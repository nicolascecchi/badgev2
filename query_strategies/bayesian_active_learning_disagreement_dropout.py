import numpy as np
import torch
from .strategy import Strategy
from torch.utils.data import DataLoader

class BALDDropout(Strategy):
	def __init__(self, X, Y, idxs_lb, idxs_val, net, handler, args, n_drop=10):
		super(BALDDropout, self).__init__(X, Y, idxs_lb, idxs_val, net, handler, args)
		self.n_drop = n_drop

	def query(self, n):
		idxs_unlabeled = self.availQuery
		probs = self.predict_prob_dropout_split(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled], self.n_drop)
		pb = probs.mean(0)
		entropy1 = (-pb*torch.log(pb)).sum(1)
		entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
		U = entropy2 - entropy1
		return idxs_unlabeled[U.sort()[1][:n]]
        
	def predict_prob_dropout_split(self, X, Y, n_drop):
		loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),shuffle=False, **self.args['loader_te_args'
		self.clf.train()
		probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
		with torch.no_grad():
		    for i in range(n_drop):
		        print('n_drop {}/{}'.format(i+1, n_drop))
	        	for x, y, idxs in loader_te:
		            x, y = Variable(x.cuda()), Variable(y.cuda())
	            	out, e1 = self.clf(x)
	            	probs[i][idxs] += F.softmax(out, dim=1).cpu().data
	    probs /= n_drop
		return probs