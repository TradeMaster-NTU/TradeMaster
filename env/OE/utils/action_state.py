import numpy as np
from env.OE.utils.feature_type import FeatureType

class ActionState(object):

    def __init__(self, t, i, reference_price, market={}):
        self.t = t
        self.i = i
        self.reference_price = reference_price
        self.market = market

    def __hash__(self):
        return hash((self.t, self.i, self.reference_price, frozenset(self.market.items())))

    def __eq__(self, other):
        return (self.t, self.i, self.reference_price, frozenset(self.market.items())) == (other.t, other.i, other.reference_price, frozenset(self.market.items()))

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)

    def __str__(self):
        return str((self.t, self.i, self.reference_price, str(self.market)))

    def __repr__(self):
        return self.__str__()

    def toArray(self):
        if FeatureType.ORDERS.value in self.market:
            # arr = [np.array([self.getT()]), np.array([self.getI()])]
            # for k, v in self.getMarket().items():
            #     arr.append(v)
            # return np.array([arr])
            features = self.market[FeatureType.ORDERS.value]
            mean1d = self.market.get('mean1d', 0)
            arr = np.zeros(shape=(1,features.shape[1],2), dtype=float)
            arr[0,0] = np.array([self.t, self.i])
            arr[0,1] = np.array([self.reference_price, mean1d])
            features = np.vstack((arr, features))
            #return features.reshape((1, features.shape[0], features.shape[1], features.shape[2])) # required for custom DQN
            return features.reshape((features.shape[0], features.shape[1], features.shape[2])) # required for baseline DQN
            #return features # (2*lookback, levels, count(features))
        elif FeatureType.TRADES.value in self.market:
            features = self.market[FeatureType.TRADES.value]
            features = np.vstack((np.array([self.t, self.i, 0]), features))
            return features
        else:
            Exception("Feature not known to ActionState.")

    def getT(self):
        return self.t

    def setT(self, t):
        self.t = t

    def getI(self):
        return self.i

    def setI(self, i):
        self.i = i

    def getMarket(self):
        return self.market
