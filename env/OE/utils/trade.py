from datetime import datetime

""" A trade is an extended version of a Position, indicating the purchases made """
class Trade:
    def __init__(self, orderSide, orderType, cty, price, fee=0.0, timestamp=str(datetime.now()).split('.')[0]):
        self.orderSide = orderSide
        self.orderType = orderType
        self.cty = cty
        self.price = price
        self.fee = fee
        self.timestamp = timestamp

    def __str__(self):
        return (str(self.timestamp) + ',' +
                str(self.getSide()) + ',' +
                str(self.getType()) + ',' +
                str(self.getCty()) + ',' +
                str(self.getPrice()) + ',' +
                str(self.getFee()))

    def __repr__(self):
        return str(self)

    def getSide(self):
        return self.orderSide

    def getType(self):
        return self.orderType

    def getCty(self):
        return self.cty

    def setCty(self, cty):
        self.cty = cty

    def getPrice(self):
        return self.price

    def getFee(self):
        return self.fee

    def getTimeStamp(self):
        return self.timestamp
