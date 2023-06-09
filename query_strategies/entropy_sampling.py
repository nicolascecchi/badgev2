import numpy as np
import torch
from .strategy import Strategy

class EntropySampling(Strategy):
	def __init__(self, X, Y, idxs_lb, idxs_val, net, handler, args):
		super(EntropySampling, self).__init__(X, Y, idxs_lb, idxs_val, net, handler, args)

	def query(self, n):
		idxs_unlabeled = self.availQuery
		probs = self.predict_prob(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)
		return idxs_unlabeled[U.sort()[1][:n]]
