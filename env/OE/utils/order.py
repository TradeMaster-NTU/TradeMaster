from datetime import datetime
from env.OE.utils.order_type import OrderType


class Order:
    """
    An order indicates the purchases to be made.

    The MatchEngine will try to execute an order given orders from other
    parties. Therefore, an order may result in one or many Trades.
    """

    def __init__(self, orderType, orderSide, cty, price=None, timestamp=None):
        self.timestamp = timestamp
        if not self.timestamp:
            self.timestamp = str(datetime.now()).split('.')[0]
        self.orderType = orderType
        self.orderSide = orderSide
        self.cty = cty
        self.price = price  # None for OrderSide.MARKET
        self.timestamp = timestamp
        if self.orderType == OrderType.MARKET and self.price is not None:
            raise Exception('Market order must not have a price.')
        if self.orderType == OrderType.LIMIT and self.price is None:
            raise Exception('Limit order must have a price.')

    def __str__(self):
        return (str(self.timestamp) + ',' +
                str(self.getType()) + ',' +
                str(self.getCty()) + ',' +
                str(self.getPrice()))

    def __repr__(self):
        return str(self)

    def getType(self):
        return self.orderType

    def setType(self, type):
        self.orderType = type

    def getSide(self):
        return self.orderSide

    def getCty(self):
        return self.cty

    def setCty(self, cty):
        self.cty = cty

    def getPrice(self):
        return self.price

    def setPrice(self, price):
        self.price = price

    def getTimeStamp(self):
        return self.timestamp
