import logging
import copy
import random
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from utils.action import Action
from utils.action_state import ActionState
from utils.order import Order
from utils.order_type import OrderType
from utils.order_side import OrderSide
from utils.feature_type import FeatureType

import os
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection

plt.rcParams.update({'font.size': 20})
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


class ExecutionEnv_TwoActions(gym.Env):

    def __init__(self):
        self.orderbookIndex = None
        self.actionState = None
        self.execution = None
        self.episode = 0
        self._configure()
        self.done = False
        self.initOrderbookIndex = 0

    def _generate_Sequence(self, min, max, step):
        """ Generate sequence (that unlike xrange supports float)

        max: defines the sequence maximum
        step: defines the interval
        """
        i = min
        I = []
        while i <= max:
            I.append(i)
            i = i + step
        return I

    def _configure(self,
                   orderbook=None,
                   side=OrderSide.SELL,
                   levels=(-50, 50, 1),
                   T=(0, 100, 10),
                   I=(0, 1, 0.1),
                   lookback=25,
                   bookSize=5,
                   featureType=FeatureType.ORDERS,
                   callbacks=[]
                   ):
        self.orderbook = orderbook
        self.side = side
        self.levels = self._generate_Sequence(min=levels[0], max=levels[1], step=levels[2])
        self.action_space = spaces.Discrete(len(self.levels))
        self.T = self._generate_Sequence(min=T[0], max=T[1], step=T[2])
        self.I = self._generate_Sequence(min=I[0], max=I[1], step=I[2])
        self.lookback = lookback  # results in (bid|size, ask|size) -> 4*5
        self.bookSize = bookSize
        self.featureType = featureType
        if self.featureType == FeatureType.ORDERS:
            self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(2 * self.lookback + 1, self.bookSize, 2))
        else:
            self.observation_space = spaces.Box(low=0.0, high=100.0, shape=(self.lookback + 1, 3))
        self.callbacks = callbacks
        self.episodeActions = []
        self.inventoryLevels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def setOrderbook(self, orderbook):
        self.orderbook = orderbook

    def setSide(self, side):
        self.side = side

    def setLevels(self, min, max, step):
        self.levels = self._generate_Sequence(min=min, max=max, step=step)
        self.action_space = spaces.Discrete(len(self.levels))

    def setT(self, min, max, step):
        self.T = self._generate_Sequence(min=min, max=max, step=step)

    def setI(self, min, max, step):
        self.I = self._generate_Sequence(min=min, max=max, step=step)

    def setLookback(self, lookback):
        self.lookback = lookback
        if self.bookSize is not None:
            self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(2 * self.lookback, self.bookSize, 2))

    def setBookSize(self, bookSize):
        self.bookSize = bookSize
        if self.lookback is not None:
            self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(2 * self.lookback, self.bookSize, 2))

    def _determine_next_inventory(self, execution):
        qty_remaining = execution.getQtyNotExecuted()
        # TODO: Working with floats requires such an ugly threshold
        if qty_remaining > 0.0000001:
            # Approximate next closest inventory given remaining and I
            i_next = min([0.0] + self.I, key=lambda x: abs(x - qty_remaining))
            logging.info('Qty remain: ' + str(qty_remaining)
                         + ' -> inventory: ' + str(qty_remaining)
                         + ' -> next i: ' + str(i_next))
        else:
            i_next = 0.0

        logging.info('Next inventory for execution: ' + str(i_next))
        return i_next

    def _determine_next_time(self, t):
        if t > 0:
            t_next = self.T[self.T.index(t) - 1]
        else:
            t_next = t

        logging.info('Next timestep for execution: ' + str(t_next))
        return t_next

    def _determine_runtime(self, t):
        if t != 0:
            T_index = self.T.index(t)
            runtime = self.T[T_index] - self.T[T_index - 1]
        else:
            runtime = t
        return runtime

    def _get_random_orderbook_state(self):
        return self.orderbook.getRandomStateInDay(runtime=max(self.T), min_head=self.lookback)

    def _create_execution(self, a, inventory):
        runtime = self._determine_runtime(self.actionState.getT())
        orderbookState = self.orderbook.getState(self.orderbookIndex)
        self.totalOrderInventory = self.actionState.getI()

        if runtime <= 0.0 or a is None:
            price = None
            ot = OrderType.MARKET
            currentOrderInventory = self.actionState.getI()
        else:
            if a == 'skip':
                currentOrderInventory = 0
                price = orderbookState.getPriceAtLevel(self.side, 0)
            else:
                # currentOrderInventory = self.actionState.getI()
                # currentOrderInventory = int(self.totalOrderInventory / 10)
                currentOrderInventory = int(inventory * self.totalOrderInventory / self.inventoryLevels[-1])
                price = orderbookState.getPriceAtLevel(self.side, a)
            ot = OrderType.LIMIT

        # print("price: {}".format(price))
        order = Order(
            orderType=ot,
            orderSide=self.side,
            cty=currentOrderInventory,
            price=price
        )
        execution = Action(a=a, runtime=runtime)
        execution.setTotalInventory(self.totalOrderInventory)
        execution.setState(self.actionState)
        execution.setOrder(order)
        execution.setOrderbookState(orderbookState)
        execution.setOrderbookIndex(self.orderbookIndex)
        execution.setReferencePrice(orderbookState.getBestAsk())
        return execution

    def _update_execution(self, execution, a, inventory):
        runtime = self._determine_runtime(self.actionState.getT())
        orderbookState = self.orderbook.getState(self.orderbookIndex)

        if runtime <= 0.0 or a is None:
            price = None
            ot = OrderType.MARKET
            currentOrderInventory = self.actionState.getI()
        else:
            if a == 'skip':
                currentOrderInventory = 0
                price = execution.getOrderbookState().getPriceAtLevel(self.side, 0)
            else:
                # currentOrderInventory = self.actionState.getI()
                # currentOrderInventory = min(self.actionState.getI(), self.totalOrderInventory/10)
                currentOrderInventory = min(self.actionState.getI(),
                                            int(inventory * self.totalOrderInventory / self.inventoryLevels[-1]))
                price = execution.getOrderbookState().getPriceAtLevel(self.side, a)
            ot = OrderType.LIMIT

        # print("price: {}".format(price))
        order = Order(
            orderType=ot,
            orderSide=self.side,
            cty=currentOrderInventory,
            price=price
        )
        execution.setRuntime(runtime)
        execution.setState(self.actionState)
        execution.setOrder(order)
        execution.setOrderbookState(orderbookState)
        execution.setOrderbookIndex(self.orderbookIndex)
        return execution

    def _makeFeature(self, orderbookIndex, qty, reference_price):
        if self.featureType == FeatureType.ORDERS:
            return self.orderbook.getBidAskFeatures(
                state_index=orderbookIndex,
                lookback=self.lookback,
                reference_price=reference_price,
                qty=self.I[-1],  # i_next+0.0001,
                normalize=True,
                price=True,
                size=True,
                levels=self.bookSize
            )
        else:
            state = self.orderbook.getState(orderbookIndex)
            return self.orderbook.getHistTradesFeature(
                ts=state.getUnixTimestamp(),
                lookback=self.lookback,
                normalize=False,
                norm_size=qty,
                norm_price=state.getBidAskMid()
            )

    def step(self, action, inventory):
        self.episode += 1
        if action == 0:
            action = 'skip'
        else:
            action = self.levels[action]
        self.episodeActions.append(action)
        if self.execution is None:
            self.execution = self._create_execution(action, inventory)
        else:
            self.execution = self._update_execution(self.execution, action, inventory)

        logging.info(
            'Created/Updated execution.' +
            '\nAction: ' + str(action) + ' (' + str(self.execution.getOrder().getType()) + ')' +
            '\nt: ' + str(self.actionState.getT()) +
            '\nruntime: ' + str(self.execution.getRuntime()) +
            '\ni: ' + str(self.actionState.getI())
        )
        self.execution, counterTrades = self.execution.run(self.orderbook)

        i_next = self._determine_next_inventory(self.execution)
        t_next = self._determine_next_time(self.execution.getState().getT())

        feature = self._makeFeature(orderbookIndex=self.execution.getOrderbookIndex(), qty=i_next,
                                    reference_price=self.execution.getReferencePrice())
        # mean1d = self.orderbook.getState(self.execution.getOrderbookIndex()).market['mean1d']
        # state_next = ActionState(t_next, i_next, self.execution.getReferencePrice(),
        #                          {self.featureType.value: feature, 'mean1d': mean1d})
        state_next = ActionState(t_next, i_next, self.execution.getReferencePrice(),
                                 {self.featureType.value: feature})
        done = self.execution.isFilled() or state_next.getI() == 0
        self.done = done
        if done:
            reward = self.execution.getReward()
            volumeRatio = 1.0
            if self.callbacks is not []:
                for cb in self.callbacks:
                    cb.on_episode_end(self.episode, {'episode_reward': reward, 'episode_actions': self.episodeActions})
            self.episodeActions = []
        else:
            reward, volumeRatio = self.execution.calculateRewardWeighted(counterTrades, self.I[-1])

        logging.info(
            'Run execution.' +
            '\nTrades: ' + str(len(counterTrades)) +
            '\nReward: ' + str(reward) + ' (Ratio: ' + str(volumeRatio) + ')' +
            '\nDone: ' + str(done)
        )
        self.orderbookIndex = self.execution.getOrderbookIndex()
        self.actionState = state_next
        return state_next.toArray(), reward, done, {}

    def reset(self):
        return self._reset(t=self.T[-1], i=self.I[-1])

    def _reset(self, t, i):
        orderbookState, orderbookIndex = self._get_random_orderbook_state()
        feature = self._makeFeature(orderbookIndex=orderbookIndex, qty=i, reference_price=orderbookState.getBestAsk())
        state = ActionState(t, i, orderbookState.getBestAsk(), {self.featureType.value: feature})  # np.array([[t, i]])
        self.execution = None
        self.orderbookIndex = orderbookIndex
        self.initOrderbookIndex = orderbookIndex
        self.actionState = state
        return state.toArray()

    def render(self, mode='human', close=False):
        list_of_datetimes = []
        values = []
        preTimestamp = None
        for i in range(-40, 80):
            lastOrderbookState = self.orderbook.getState(self.initOrderbookIndex + i)
            list_of_datetimes.append(lastOrderbookState.timestamp)
            values.append(lastOrderbookState.tradePrice)
        difference = [(list_of_datetimes[i + 1] - date).total_seconds() for (i, date) in enumerate(list_of_datetimes) if
                      i < len(list_of_datetimes) - 1]

        if self.done:
            trades = self.execution.getTrades()
            reward = self.execution.calculateReward(trades)
            direction = "Buy" if trades[0].orderSide == OrderSide.BUY else "Sell"
            delta_tradeTime = [(trade.timestamp - list_of_datetimes[0]).total_seconds() for trade in trades]
            trade_x = []
            for delta in delta_tradeTime:
                for i in range(len(difference)):
                    if delta == sum(difference[:i]):
                        trade_x.append(i + 1)
            tradeTime = [trade.timestamp.strftime("%H:%M:%S") for trade in trades]
            tradePrice = [trade.price for trade in trades]
            text = [[int(trade.cty), '{0:.2f}'.format(trade.price)] for trade in trades]

            fig = plt.figure(figsize=(6, 4.5))

            x_len = len(list_of_datetimes)
            x = np.array(list(range(x_len)))
            values = np.array(values)

            points = np.array([x, values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create line segments: 1--2, 2--17, 17--20, 20--16, 16--3, etc.
            segments_x = np.r_[x[0], x[1:-1].repeat(2), x[-1]].reshape(-1, 2)
            segments_y = np.r_[values[0], values[1:-1].repeat(2), values[-1]].reshape(-1, 2)

            linecolors = ['c' if x_[1] < 40 or x_[0] >= 100 else 'blue'
                          for x_ in segments_x]
            plt.gca().add_collection(LineCollection(segments, colors=linecolors))
            plt.xlim(x.min(), x.max())
            plt.ylim(values.min() - 0.05, values.max() + 0.05)

            frequency = 20
            test_dates = [date.strftime("%H:%M:%S") for date in list_of_datetimes]
            plt.xticks(list(range(0, x_len, frequency)), test_dates[:x_len:frequency], rotation=20)
            plt.yticks(np.arange(min(values) - 0.05, max(values) + 0.05, 0.1))
            # marker = "+" if direction == "Buy" else '.'
            sca = plt.scatter(trade_x, tradePrice, c='red', marker='x', s=20)
            # for i, (cty,price) in enumerate(text):
            # plt.annotate("{}".format(i), (trade_x[i], tradePrice[i]), xytext=(trade_x[i]-2, tradePrice[i]-0.01*i))
            # plt.annotate("({},{})".format(cty,round(price, 2)), (trade_x[i], tradePrice[i]),
            #              xytext=(trade_x[i]-2, tradePrice[i]-0.01*i),arrowprops=dict(arrowstyle="->",connectionstyle="angle3,angleA=0,angleB=-90"))
            table = plt.table(cellText=text,
                              colWidths=[0.2, 0.2],
                              rowLabels=tradeTime,
                              colLabels=['Quantity', 'Price'],
                              loc='lower right')
            table.auto_set_font_size(False)
            table.set_fontsize(13)
            table.scale(1, 1.5)
            plt.legend()
            plt.legend((sca,), (direction,), scatterpoints=1, loc='upper left')
            plt.ylabel('Stock Price')
            plt.xlabel("Trading Time")
            plt.grid(True)
            # plt.savefig(PATH_prefix + 'backtest.png')
            plt.show()
            p1 = PdfPages('backtest_{}.pdf'.format(reward))
            p1.savefig(fig)
            p1.close()
            # print(self.execution.getTrades())
        else:
            # print(self.orderbook.getState(self.initOrderbookIndex+i))
            pass

    def seed(self, seed):
        pass