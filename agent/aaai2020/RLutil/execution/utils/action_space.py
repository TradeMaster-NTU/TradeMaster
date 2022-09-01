import logging
import numpy as np
from action import Action
from order import Order
from order_type import OrderType
from order_side import OrderSide
from qlearn import QLearn
from action_state import ActionState


class ActionSpace(object):
    """DEPRECATED: use ctc-executioner-v0 instead.

    This class still contains some logic which was moved to the agent.
    """

    def __init__(self, orderbook, side, T, I, ai=None, levels=None):
        self.orderbook = orderbook
        self.side = side
        self.levels = levels
        if not ai:
            ai = QLearn(self.levels)  # levels are our qlearn actions
        self.ai = ai
        self.T = T
        self.I = I

    def getRandomOrderbookState(self):
        return self.orderbook.getRandomState(max(self.T))

    def createAction(self, level, state, orderbookIndex=None, force_execution=False):
        # Determines whether to run and force execution of given t, or if
        # segmentation of t into multiple runtimes is allowed.
        if force_execution:
            runtime = state.getT()
            ot = OrderType.LIMIT_T_MARKET
        else:
            runtime = self.determineRuntime(state.getT())
            ot = OrderType.LIMIT

        if orderbookIndex is None:
            orderbookState, orderbookIndex = self.getRandomOrderbookState()
        else:
            orderbookState = self.orderbook.getState(orderbookIndex)

        if runtime <= 0.0 or level is None:
            price = None
            ot = OrderType.MARKET
        else:
            price = orderbookState.getPriceAtLevel(self.side, level)

        order = Order(
            orderType=ot,
            orderSide=self.side,
            cty=state.getI(),
            price=price
        )
        action = Action(a=level, runtime=runtime)
        action.setState(state)
        action.setOrder(order)
        action.setOrderbookState(orderbookState)
        action.setOrderbookIndex(orderbookIndex)
        action.setReferencePrice(orderbookState.getBestAsk())
        return action

    def updateAction(self, action, level, state, orderbookIndex=None, force_execution=False):
        if force_execution:
            runtime = state.getT()
            ot = OrderType.LIMIT_T_MARKET
        else:
            runtime = self.determineRuntime(state.getT())
            ot = OrderType.LIMIT

        if orderbookIndex is not None:
            orderbookState = self.orderbook.getState(orderbookIndex)
            action.setOrderbookState(orderbookState)
            action.setOrderbookIndex(orderbookIndex)

        if runtime <= 0.0 or level is None:
            price = None
            ot = OrderType.MARKET
        else:
            price = action.getOrderbookState().getPriceAtLevel(self.side, level)

        order = Order(
            orderType=ot,
            orderSide=self.side,
            cty=state.getI(),
            price=price
        )
        action.setState(state)
        action.setOrder(order)
        return action

    def createActions(self, runtime, qty, force_execution=False):
        actions = []
        for level in self.levels:
            actions.append(self.createAction(level, runtime, qty, force_execution))
        return actions

    def determineBestAction(self, actions):
        bestAction = None
        for action in actions:
            if not bestAction:
                bestAction = action
                continue
            if action.getReward() < bestAction.getReward():
                bestAction = action
        return bestAction

    def determineRuntime(self, t):
        if t != 0:
            T_index = self.T.index(t)
            runtime = self.T[T_index] - self.T[T_index - 1]
        else:
            runtime = t
        return runtime

    def determineNextTime(self, t):
        if t > 0:
            t_next = self.T[self.T.index(t) - 1]
        else:
            t_next = t

        logging.info('Next timestep for action: ' + str(t_next))
        return t_next

    def determineNextInventory(self, action):
        qty_remaining = action.getQtyNotExecuted()

        # TODO: Working with floats requires such an ugly threshold
        if qty_remaining > 0.0000001:
            # Approximate next closest inventory given remaining and I
            i_next = min([0.0] + self.I, key=lambda x: abs(x - qty_remaining))
            logging.info('Qty remain: ' + str(qty_remaining)
                         + ' -> inventory: ' + str(qty_remaining)
                         + ' -> next i: ' + str(i_next))
        else:
            i_next = 0.0

        logging.info('Next inventory for action: ' + str(i_next))
        return i_next

    def update(self, t, i, force_execution=False):
        aiState = ActionState(t, i)
        a = self.ai.chooseAction(aiState)
        # print('Random action: ' + str(level) + ' for state: ' + str(aiState))
        action = self.createAction(level=a, state=aiState, force_execution=force_execution)
        action, counterTrades = action.run(self.orderbook)
        i_next = self.determineNextInventory(action)
        t_next = self.determineNextTime(t)
        reward = action.getReward()
        # reward = action.getValueExecuted()
        # reward = action.getTestReward()
        state_next = ActionState(action.getState().getT(), action.getState().getI(), action.getState().getMarket())
        state_next.setT(t_next)
        state_next.setI(i_next)
        #print("Reward " + str(reward) + ": " + str(action.getState()) + " with " + str(action.getA()) + " -> " + str(state_next))
        self.ai.learn(
            state1=action.getState(),
            action1=action.getA(),
            reward=reward,
            state2=state_next
        )
        return (t_next, i_next)


    def train(self, episodes=1, force_execution=False):
        for episode in range(int(episodes)):
            for t in self.T:
                logging.info("\n"+"t=="+str(t))
                for i in self.I:
                    logging.info("     i=="+str(i))
                    logging.info("Action run " + str((t, i)))
                    (t_next, i_next) = self.update(t, i, force_execution)
                    while i_next != 0:
                        if force_execution:
                            raise Exception("Enforced execution left " + str(i_next) + " unexecuted.")
                        logging.info("Action transition " + str((t, i)) + " -> " + str((t_next, i_next)))
                        (t_next, i_next) = self.update(t_next, i_next, force_execution)


    def backtest(self, q=None, episodes=10, average=False, fixed_a=None):
        if q is None:
            q = self.ai.q
        else:
            self.ai.q = q

        if not q:
            raise Exception('Q-Table is empty, please train first.')

        Ms = []
        #T = self.T[1:len(self.T)]
        for t in [self.T[-1]]:
            logging.info("\n"+"t=="+str(t))
            for i in [self.I[-1]]:
                logging.info("     i=="+str(i))
                actions = []
                state = ActionState(t, i, {})
                #print(state)
                if fixed_a is not None:
                    a = fixed_a
                else:
                    try:
                        a = self.ai.getQAction(state, 0)
                        # print("Q action for state " + str(state) + ": " + str(a))
                    except:
                        # State might not be in Q-Table yet, more training requried.
                        logging.info("State " + str(state) + " not in Q-Table.")
                        break
                actions.append(a)
                action = self.createAction(level=a, state=state, force_execution=False)
                midPrice = action.getReferencePrice()

                #print("before...")
                #print(action)
                action.run(self.orderbook)
                #print("after...")
                #print(action)
                i_next = self.determineNextInventory(action)
                t_next = self.determineNextTime(t)
                # print("i_next: " + str(i_next))
                while i_next != 0:
                    state_next = ActionState(t_next, i_next, {})
                    if fixed_a is not None:
                        a_next = fixed_a
                    else:
                        try:
                            a_next = self.ai.getQAction(state_next, 0)
                            # print("t: " + str(t_next))
                            # print("i: " + str(i_next))
                            # print("Action: " + str(a_next))
                            # print("Q action for next state " + str(state_next) + ": " + str(a_next))
                        except:
                            # State might not be in Q-Table yet, more training requried.
                            # print("State " + str(state_next) + " not in Q-Table.")
                            break
                    actions.append(a_next)
                    #print("Action transition " + str((t, i)) + " -> " + str(aiState_next) + " with " + str(runtime_next) + "s runtime.")

                    runtime_next = self.determineRuntime(t_next)
                    action.setState(state_next)
                    action.update(a_next, runtime_next)
                    action.run(self.orderbook)
                    #print(action)
                    i_next = self.determineNextInventory(action)
                    t_next = self.determineNextTime(t_next)

                price = action.getAvgPrice()
                # TODO: last column is for for the BUY scenario only
                if action.getOrder().getSide() == OrderSide.BUY:
                    profit = midPrice - price
                else:
                    profit = price - midPrice
                Ms.append([state, midPrice, actions, price, profit])
        if not average:
            return Ms
        return self.averageBacktest(Ms)

    def averageBacktest(self, M):
        # Average states within M
        N = []
        observed = []
        for x in M:
            state = x[0]
            if state in observed:
                continue
            observed.append(state)
            paid = []
            reward = []
            for y in M:
                if y[0] == state:
                    paid.append(y[3])
                    reward.append(y[4])
            N.append([state, x[1], x[2], np.average(paid), np.average(reward)])
        return N
