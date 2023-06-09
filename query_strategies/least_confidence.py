import numpy as np
from .strategy import Strategy
import pdb
class LeastConfidence(Strategy):
    def __init__(self, X, Y, idxs_lb, idxs_val, net, handler, args):
        super(LeastConfidence, self).__init__(X, Y, idxs_lb, idxs_val, net, handler, args)

    def query(self, n):
        idxs_unlabeled = self.availQuery
        probs = self.predict_prob(self.X[idxs_unlabeled], np.asarray(self.Y)[idxs_unlabeled])
        U = probs.max(1)[0]
        return idxs_unlabeled[U.sort()[1][:n]]
