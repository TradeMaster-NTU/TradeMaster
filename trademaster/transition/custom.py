from .builder import TRANSITIONS
from collections import namedtuple

@TRANSITIONS.register_module()
def Transition():
    return namedtuple("Transition", ['state',
                                     'action',
                                     'reward',
                                     'undone',
                                     'next_state'])

@TRANSITIONS.register_module()
def TransitionPD():
    return namedtuple("Transition", ['state',
                                     'action',
                                     'reward',
                                     'undone',
                                     'next_state',
                                     'public_state',
                                     'private_state',
                                     'next_public_state',
                                     'next_private_state'])

@TRANSITIONS.register_module()
def TransitionDeepTrader():
    return namedtuple("TransitionDeepTrader",
                                    ['state',
                                     'action',
                                     'reward',
                                     'undone',
                                     'next_state',
                                     'correlation_matrix',
                                     'next_correlation_matrix',
                                     'state_market',
                                     'roh_bar_market'
                                     ])