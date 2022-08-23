import gym
from RLutil.execution.utils.orderbook import Orderbook
from RLutil.execution.utils.order_side import OrderSide

import RLutil.execution

def create_stock_environment(stock_name, volume=-40000):
    if volume < 0:
        side = OrderSide.SELL
        volume *= -1
    else:
        side = OrderSide.BUY
    orderbook = Orderbook(extraFeatures=False)
    orderbook.loadFromFile(stock_name)
    orderbook.generateDict()
    max_price_level=10
    # max_price_level = int(int(round(orderbook.getState(0).tradePrice * 0.005, 2)/0.01)/10)*10
    env = gym.make("ctc-executioner-v1")
    env._configure(orderbook, side=side, levels=(-max_price_level, 0, max_price_level/10), T=(0, 1800, 360), I=(0, volume, int(volume/10)), lookback=2 * 60 * 2)
    return env
