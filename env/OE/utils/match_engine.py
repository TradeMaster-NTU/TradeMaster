from env.OE.utils.trade import Trade
from env.OE.utils.order import Order
from env.OE.utils.order_type import OrderType
from env.OE.utils.order_side import OrderSide
import copy
import logging
logger = logging.getLogger(__name__)
import numpy as np

class MatchEngine(object):

    def __init__(self, orderbook, index=0, maxRuntime=100):
        self.orderbook = orderbook
        self.index = index
        self.maxRuntime = maxRuntime
        self.matches = set()
        self.recordMatches = False

    def _removePosition(self, side, price, qty):
        if self.recordMatches == True:
            self.matches.add((side, price, qty))

    def _isRemoved(self, side, price, qty):
        return (side, price, qty) in self.matches

    def setIndex(self, index):
        self.index = index

    def matchLimitOrder(self, order, orderbookState):
        """
        Attempts to match a limit Order in an order book state.

        Parameters
        ----------
        order : Order
            Order defines the will to buy or sell under certain conditions.
        orderbookState : OrderbookState
            The state of the order book to attempt matching the provided order

        Returns
        -------
        [Trades]
            A list of the resulted trades resulted during the matching process.
        """
        if order.getSide() == OrderSide.BUY:
            bookSide = orderbookState.getSellers()
        else:
            bookSide = orderbookState.getBuyers()

        def isMatchingPosition(p):
            if order.getSide() == OrderSide.BUY:
                return bookSide[sidePosition].getPrice() <= order.getPrice()
            else:
                return bookSide[sidePosition].getPrice() >= order.getPrice()

        partialTrades = []
        remaining = order.getCty()
        sidePosition = 0
        while len(bookSide) > sidePosition and isMatchingPosition(sidePosition) and remaining > 0.0:
            p = bookSide[sidePosition]
            price = p.getPrice()
            qty = p.getQty()

            # skip if position was already machted
            if self._isRemoved(side=order.getSide(), price=price, qty=qty):
                continue

            if not partialTrades and qty >= order.getCty():
                logging.debug("Full execution: " + str(qty) + " pcs available")
                t = Trade(orderSide=order.getSide(), orderType=OrderType.LIMIT, cty=remaining, price=price, timestamp=orderbookState.getTimestamp())
                #self._removePosition(side=order.getSide(), price=price, qty=qty)
                return [t]
            else:
                logging.debug("Partial execution: " + str(qty) + " pcs available")
                t = Trade(orderSide=order.getSide(), orderType=OrderType.LIMIT, cty=min(qty, remaining), price=price, timestamp=orderbookState.getTimestamp())
                partialTrades.append(t)
                #self._removePosition(side=order.getSide(), price=price, qty=qty)
                sidePosition = sidePosition + 1
                remaining = remaining - qty

                if sidePosition == len(bookSide) - 1:
                    # At this point there is no more liquidity in this state of the order
                    # book (data) but the order price might actually be still higher than
                    # what was available. For convenience sake we assume that there would
                    # be liquidity in the subsequent levels above.
                    # Therefore we linearly interpolate and place fake orders from
                    # imaginary traders in the book with an increased price (according to
                    # derivative) and similar qty.
                    average_qty = np.mean([x.getCty() for x in partialTrades])
                    logging.debug("On average executed qty: " + str(average_qty))
                    if average_qty == 0.0:
                        average_qty = 0.5
                        logging.debug("Since no trades were executed (e.g. true average executed qty == 0.0), defaul is choosen: " + str(average_qty))
                    derivative_price = abs(np.mean(np.gradient([x.getPrice() for x in partialTrades])))
                    logging.debug("Derivative of price from executed trades: " + str(derivative_price))
                    if derivative_price == 0.0:
                        derivative_price = 10.0
                        logging.debug("Since no trades were executed (e.g. derivative executed price == 0.0), defaul is choosen: " + str(derivative_price))
                    while remaining > 0.0:
                        if order.getSide() == OrderSide.BUY:
                            price = price + derivative_price
                            if price > order.getPrice():
                                break
                        elif order.getSide() == OrderSide.SELL:
                            price = price - derivative_price
                            if price < order.getPrice():
                                break

                        qty = min(average_qty, remaining)
                        logging.debug("Partial execution: assume " + str(qty) + " available")
                        partialTrades.append(Trade(orderSide=order.getSide(), orderType=OrderType.LIMIT, cty=qty, price=price, timestamp=orderbookState.getTimestamp()))
                        remaining = remaining - qty

        return partialTrades

    def matchMarketOrder(self, order, orderbookState):
        """
        Matches an within an order book state.

        Parameters
        ----------
        order : Order
            Order defines the will to buy or sell under certain conditions.
        orderbookState : OrderbookState
            The state of the order book to attempt matching the provided order

        Returns
        -------
        [Trades]
            A list of the resulted trades resulted during the matching process.
        """
        if order.getSide() == OrderSide.BUY:
            bookSide = orderbookState.getSellers()
        else:
            bookSide = orderbookState.getBuyers()

        partialTrades = []
        remaining = order.getCty()
        sidePosition = 0
        price = 0.0
        while len(bookSide) > sidePosition and remaining > 0.0:
            p = bookSide[sidePosition]
            derivative_price = p.getPrice() - price
            price = p.getPrice()
            qty = p.getQty()
            if not partialTrades and qty >= order.getCty():
                logging.debug("Full execution: " + str(qty) + " pcs available")
                return [Trade(orderSide=order.getSide(), orderType=OrderType.MARKET, cty=remaining, price=price, timestamp=orderbookState.getTimestamp())]
            else:
                logging.debug("Partial execution: " + str(qty) + " pcs available")
                qtyExecute = min(qty, remaining)
                partialTrades.append(Trade(orderSide=order.getSide(), orderType=OrderType.MARKET, cty=qtyExecute, price=price, timestamp=orderbookState.getTimestamp()))
                sidePosition = sidePosition + 1
                remaining = remaining - qtyExecute
                logging.debug("Remaining: " + str(remaining))

        return partialTrades

    def matchOrder(self, order, seconds=None):
        """
        Matches an Order according to its type.

        This function serves as the main interface for Order matching.
        Orders are being matched differently according to their OrderType.
        In addition, an optional time interval can be defines from how long the
        matching process should run and therefore simulates what is generally
        known as *Good Till Time (GTT)*.
        After the time is consumed, the order is either removed (e.g. neglected)
        in case of a standard OrderType.LIMIT or a matching on market follows in
        case OrderType.LIMIT_T_MARKET was defined.

        Parameters
        ----------
        order : Order
            Order defines the will to buy or sell under certain conditions.
        seconds : int
            Good Till Time (GTT)

        Returns
        -------
        [Trades]
            A list of the resulted trades resulted during the matching process.
        float
            Quantity of unexecuted assets.
        int
            Index of order book where matching stopped.
        """
        order = copy.deepcopy(order)  # Do not modify original order!
        i = self.index
        remaining = order.getCty()
        trades = []

        while len(self.orderbook.getStates()) - 1 > i and remaining > 0:
            orderbookState = self.orderbook.getState(i)
            logging.debug("Evaluate state " + str(i) + ":\n" + str(orderbookState))

            # Stop matching process after defined seconds are consumed
            if seconds is not None:
                t_start = self.orderbook.getState(self.index).getTimestamp()
                t_now = orderbookState.getTimestamp()
                t_delta = (t_now - t_start).total_seconds()
                logging.debug(str(t_delta) + " of " + str(seconds) + " consumed.")
                if t_delta >= seconds:
                    logging.debug("Time delta consumed, stop matching.\n")
                    break

            if order.getType() == OrderType.LIMIT:
                counterTrades = self.matchLimitOrder(order, orderbookState)
            elif order.getType() == OrderType.MARKET:
                counterTrades = self.matchMarketOrder(order, orderbookState)
            elif order.getType() == OrderType.LIMIT_T_MARKET:
                if seconds is None:
                    raise Exception(str(OrderType.LIMIT_T_MARKET) + ' requires a time limit.')
                counterTrades = self.matchLimitOrder(order, orderbookState)
            else:
                raise Exception('Order type not known or not implemented yet.')

            if counterTrades:
                trades = trades + counterTrades
                logging.debug("Trades executed:")
                for counterTrade in counterTrades:
                    logging.debug(counterTrade)
                    remaining = remaining - counterTrade.getCty()
                order.setCty(remaining)
                logging.debug("Remaining: " + str(remaining) + "\n")
            else:
                logging.debug("No orders matched.\n")
            i = i + 1

        # Execute remaining qty as market if LIMIT_T_MARKET
        if remaining > 0.0 and (order.getType() == OrderType.LIMIT_T_MARKET or order.getType() == OrderType.MARKET):
            logging.debug('Execute remaining as MARKET order.')
            #i = i - 1  # back to previous state
            if not len(self.orderbook.getStates()) > i:
                raise Exception('Not enough data for following market order.')

            orderbookState = self.orderbook.getState(i)
            logging.debug("Evaluate state " + str(i) + ":\n" + str(orderbookState))
            counterTrades = self.matchMarketOrder(order, orderbookState)
            if not counterTrades:
                raise Exception('Remaining market order matching failed.')
            trades = trades + counterTrades
            logging.debug("Trades executed:")
            for counterTrade in counterTrades:
                logging.debug(counterTrade)
                remaining = remaining - counterTrade.getCty()
            order.setCty(remaining)
            logging.debug("Remaining: " + str(remaining) + "\n")

        logging.debug("Total number of trades: " + str(len(trades)))
        logging.debug("Remaining qty of order: " + str(remaining))
        logging.debug("Index at end of match period: " + str(i))
        return trades, remaining, i


# logging.basicConfig(level=logging.DEBUG)
# from orderbook import Orderbook
# orderbook = Orderbook(extraFeatures=False)
# orderbook.loadFromFile('query_result_small.tsv')
# engine = MatchEngine(orderbook, index=0)
#
# #order = Order(orderType=OrderType.LIMIT, orderSide=OrderSide.BUY, cty=11.0, price=16559.0)
# #order = Order(orderType=OrderType.MARKET, orderSide=OrderSide.BUY, cty=25.5, price=None)
# order = Order(orderType=OrderType.LIMIT_T_MARKET, orderSide=OrderSide.SELL, cty=1.0, price=16559.0)
# trades, remaining, i = engine.matchOrder(order, seconds=1.0)
# c = 0.0
# for trade in trades:
#     c = c + trade.getCty()
# print(c)
