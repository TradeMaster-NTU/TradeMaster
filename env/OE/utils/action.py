import copy
from env.OE.utils.order_side import OrderSide
from env.OE.utils.order_type import OrderType
from env.OE.utils.match_engine import MatchEngine
import numpy as np

class Action(object):

    def __init__(self, a, runtime):
        self.a = a
        self.runtime = runtime
        self.order = None
        self.trades = []  # filled order
        self.orderbookState = None
        self.orderbookIndex = None
        self.state = None
        self.referencePrice = None
        self.totalInventory = None

    def __str__(self):
        s = '----------ACTION----------\n'
        s = s + 'Level: ' + str(self.a) + '\n'
        s = s + 'Runtime: ' + str(self.runtime) + '\n'
        s = s + 'State: ' + str(self.state) + '\n'
        s = s + 'Order: ' + str(self.order) + '\n'
        s = s + 'Reference Price: ' + str(self.referencePrice) + '\n'
        s = s + 'Book index: ' + str(self.orderbookIndex) + '\n'
        s = s + 'Book state: \n' + str(self.orderbookState) + '\n'
        s = s + '----------ACTION----------\n'
        return s

    def __repr__(self):
        return self.__str__()

    def getA(self):
        return self.a

    def setA(self, a):
        self.a = a

    def getRuntime(self):
        return self.runtime

    def setRuntime(self, runtime):
        self.runtime = runtime

    def getState(self):
        return self.state

    def setState(self, state):
        self.state = state

    def setOrderbookState(self, state):
        self.orderbookState = state

    def getOrderbookState(self):
        return self.orderbookState

    def setOrderbookIndex(self, index):
        self.orderbookIndex = index

    def getOrderbookIndex(self):
        return self.orderbookIndex

    def getReferencePrice(self):
        return self.referencePrice

    def setReferencePrice(self, referencePrice):
        self.referencePrice = referencePrice

    def getOrder(self):
        return self.order

    def setOrder(self, order):
        self.order = order

    def getTrades(self):
        return self.trades

    def setTrades(self, trades):
        self.trades = trades

    def getTotalInventory(self):
        return self.totalInventory

    def setTotalInventory(self, inventory):
        self.totalInventory = inventory

    def getAvgPrice(self):
        return self.calculateAvgPrice(self.getTrades())

    def calculateAvgPrice(self, trades):
        """Returns the average price paid for the executed order."""
        if self.calculateQtyExecuted(trades) == 0:
            return 0.0

        price = 0.0
        for trade in trades:
            price = price + trade.getCty() * trade.getPrice()
        return price / self.calculateQtyExecuted(trades)

    def getQtyExecuted(self):
        return self.calculateQtyExecuted(self.getTrades())

    def calculateQtyExecuted(self, trades):
        qty = 0.0
        for trade in trades:
            qty = qty + trade.getCty()
        return qty

    def getQtyNotExecuted(self):
        return self.getTotalInventory() - self.getQtyExecuted()

    def isFilled(self):
        return self.getQtyExecuted() == self.getTotalInventory()

    def getTotalPaidReceived(self):
        return self.getAvgPrice() * self.getQtyExecuted()

    def getReward(self):
        return self.calculateReward(self.getTrades())

    @DeprecationWarning
    def getValueAvg(self):
        return self.getReward()

    def calculateReward(self, trades):
        """Retuns difference of the average paid price to bid/ask-mid price.
        The higher, the better,
        For BUY: total paid at mid price - total paid
        For SELL: total received - total received at mid price
        """
        # In case of no executed trade, the value is the negative reference
        if self.calculateQtyExecuted(trades) == 0.0:
            return 0.0

        if self.getOrder().getSide() == OrderSide.BUY:
            reward = self.getReferencePrice() - self.calculateAvgPrice(trades)
        else:
            reward = self.calculateAvgPrice(trades) - self.getReferencePrice()

        return reward

    def calculateRewardWeighted(self, trades, inventory):
        reward = self.calculateReward(trades)
        if reward == 0.0:
            return reward, 0.0

        volumeExecuted = self.calculateQtyExecuted(trades)
        volumeRatio = volumeExecuted / inventory
        rewardWeighted = reward * volumeRatio
        return rewardWeighted, volumeRatio

    def getPcFilled(self):
        return 100 * (self.getQtyExecuted() / self.getOrder().getCty())

    def update(self, a, runtime):
        """Updates an action to be ready for the next run."""
        if runtime <= 0.0:
            price = None
            self.getOrder().setType(OrderType.MARKET)
        else:
            price = self.getOrderbookState().getPriceAtLevel(self.getOrder().getSide(), a)

        self.getOrder().setPrice(price)
        self.getOrder().setCty(self.getQtyNotExecuted())
        self.setRuntime(runtime)
        return self

    def getMatchEngine(self, orderbook):
        return MatchEngine(orderbook, self.getOrderbookIndex())

    def run(self, orderbook):
        """Runs action using match engine.
        The orderbook is provided and being used in the match engine along with
        the prviously determined index where the action should start matching.
        The matching process returns the trades and the remaining quantity
        along with the index the matching stopped.
        The action gets updated with those values accordingly such that it can
        be evaluated or run over again (e.g. with a new runtime).
        """
        matchEngine = self.getMatchEngine(orderbook)
        counterTrades, qtyRemain, index = matchEngine.matchOrder(self.getOrder(), self.getRuntime())
        self.setTrades(self.getTrades() + counterTrades) # appends trades!
        #self.setTrades(counterTrades) # only current trades!
        self.setOrderbookIndex(index=index)
        self.setOrderbookState(orderbook.getState(index))
        return self, counterTrades
