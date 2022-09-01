from enum import Enum


class OrderSide(Enum):
    BUY = 'buy'
    SELL = 'sell'

    def opposite(self):
        if self == OrderSide.BUY:
            return OrderSide.SELL
        elif self == OrderSide.SELL:
            return OrderSide.BUY
