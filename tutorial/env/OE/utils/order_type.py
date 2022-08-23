from enum import Enum


class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    LIMIT_T_MARKET = 'limit_t_market'
    CANCEL = 'cancel'
