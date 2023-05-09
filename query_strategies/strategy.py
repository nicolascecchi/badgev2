import numpy as np
from torch import nn
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
import pdb
import resnet
from torch.distributions.categorical import Categorical

class Strategy:
    def __init__(self, X, Y, idxs_lb, idxs_val, net, handler, args):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.idxs_val = idxs_val
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        use_cuda = torch.cuda.is_available()
        self.availQuery = np.arange(self.n_pool)[~(idxs_lb | idxs_val)]

    def query(self, n):
        pass

    def update(self, idxs_lb, idxs_val):
        self.idxs_lb = idxs_lb
        self.idxs_val = idxs_val
        self.availQuery = np.arange(self.n_pool)[~(self.idxs_lb | self.idxs_val)]

    def _train(self, epoch, loader_tr, optimizer):
        '''
        One step of the optimization algorithm.
        '''
        self.clf.train()
        accFinal = 0.
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = Variable(x.cuda()), Variable(y.cuda())
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y)
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
            loss.backward()

            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)
            optimizer.step()
        return accFinal / len(loader_tr.dataset.X), loss.item()

    def weight_reset(m):
        newLayer = deepcopy(m)
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    def train(self, reset=False, optimizer=0, verbose=True, data=[], net=[]):
        '''
        Trains model given the current labeled dataset.
        '''
        #n_epoch = self.args['n_epoch']
        if reset:
            self.clf =  self.net.apply(self.weight_reset).cuda()
        else:
            self.clf =  self.net.cuda() 
        
        if type(net) != list: self.clf = net
        if type(optimizer) == int: optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(), transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])
        if len(data) > 0:
            loader_tr = DataLoader(self.handler(data[0], torch.Tensor(data[1]).long(), transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])
        
        idxs_val = np.arange(self.n_pool)[self.idxs_val] 
        loader_val = DataLoader(self.handler(self.X[idxs_val], torch.Tensor(self.Y.numpy()[idxs_val]).long(), transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])

        epoch = 1
        accCurrent = 0.
        bestAcc = 0.
        attempts = 0
        lastAccVal = 0.00001 # small value to avoid div by 0
        accVal = 0.
        patience = 3

        while (accVal/lastAccVal > 1): 
            lastAccVal = accVal
            accCurrent, lossCurrent = self._train(epoch, loader_tr, optimizer)
            accVal = self.eval_validation(loader_validation=loader_val)
            print('val acc: '+str(accVal), flush=True)
            if (accVal/lastAccVal > 1):
                attempts = 0
            else: 
                attempts += 1         
            epoch += 1
            if attempts > patience:
                print('Break training: patience exhausted', flush=True)
                break
            
            if verbose: 
                print(str(epoch) + ' ' + str(attempts) + ' training accuracy: ' + str(accCurrent), flush=True)
            # reset if not converging
            if (epoch % 1000 == 0) and (accCurrent < 0.2) and (self.args['modelType'] != 'linear'):
                self.clf = self.net.apply(self.weight_reset)
                optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)
            if attempts >= 50 and self.args['modelType'] == 'linear': 
                break 

    def eval_validation(self, loader_validation):
        self.clf.eval()
        accFinal = 0
        for batch_idx, (x, y, idxs) in enumerate(loader_validation):
            out, e1 = self.clf(x)
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
        return accFinal / len(loader_validation.dataset.X)

    def train_val(self, valFrac=0.1, opt='adam', verbose=False):

        if verbose: 
            print(' ',flush=True)
        if verbose: 
            print('getting validation minimizing number of epochs', flush=True)
        self.clf =  self.net.apply(self.weight_reset).cuda()
        if opt == 'adam': 
            optimizer = optim.Adam(self.clf.parameters(), lr=self.args['lr'], weight_decay=0)
        if opt == 'sgd': 
            optimizer = optim.SGD(self.clf.parameters(), lr=self.args['lr'], weight_decay=0)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        nVal = int(len(idxs_train) * valFrac)
        idxs_train = idxs_train[np.random.permutation(len(idxs_train))]
        idxs_val = idxs_train[:nVal]
        idxs_train = idxs_train[nVal:]

        loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(), transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])

        epoch = 1
        accCurrent = 0.
        bestLoss = np.inf
        attempts = 0
        ce = nn.CrossEntropyLoss()
        valTensor = torch.Tensor(self.Y.numpy()[idxs_val]).long()
        attemptThresh = 10
        while True:
            accCurrent, lossCurrent = self._train(epoch, loader_tr, optimizer)
            valPreds = self.predict_prob(self.X[idxs_val], valTensor, exp=False)
            loss = ce(valPreds, valTensor).item()
            if loss < bestLoss:
                bestLoss = loss
                attempts = 0
                bestEpoch = epoch
            else:
                attempts += 1
                if attempts == attemptThresh: break
            if verbose: print(epoch, attempts, loss, bestEpoch, bestLoss, flush=True)
            epoch += 1

        return bestEpoch

    def get_dist(self, epochs, nEns=1, opt='adam', verbose=False):

        if verbose: print(' ',flush=True)
        if verbose: print('training to indicated number of epochs', flush=True)

        ce = nn.CrossEntropyLoss()
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(), transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])
        dataSize = len(idxs_train)        
        N = np.round((epochs * len(loader_tr)) ** 0.5)
        allAvs = []
        allWeights = []
        for m in range(nEns):

            # initialize new model and optimizer
            net =  self.net.apply(self.weight_reset).cuda()
            if opt == 'adam': optimizer = optim.Adam(net.parameters(), lr=self.args['lr'], weight_decay=0)
            if opt == 'sgd': optimizer = optim.SGD(net.parameters(), lr=self.args['lr'], weight_decay=0)
        
            nUpdates = k = 0
            ek = (k + 1) * N
            pVec = torch.cat([torch.zeros_like(p).cpu().flatten() for p in self.clf.parameters()])

            avIterates = []
            for epoch in range(epochs + 1):
                correct = lossTrain = 0.
                net = net.train()
                for ind, (x, y, _) in enumerate(loader_tr):
                    x, y = x.cuda(), y.cuda()
                    optimizer.zero_grad()
                    output, _ = net(x)
                    correct += torch.sum(output.argmax(1) == y).item()
                    loss = ce(output, y)
                    loss.backward()
                    lossTrain += loss.item() * len(y)
                    optimizer.step()
                    flat = torch.cat([deepcopy(p.detach().cpu()).flatten() for p in net.parameters()])
                    pVec = pVec + flat
                    nUpdates += 1
                    if nUpdates > ek:
                        avIterates.append(pVec / N)
                        pVec = torch.cat([torch.zeros_like(p).cpu().flatten() for p in net.parameters()])
                        k += 1
                        ek = (k + 1) * N

                lossTrain /= dataSize
                accuracy = correct / dataSize
                if verbose: print(m, epoch, nUpdates, accuracy, lossTrain, flush=True)
            allAvs.append(avIterates)
            allWeights.append(torch.cat([deepcopy(p.detach().cpu()).flatten() for p in net.parameters()]))

        for m in range(nEns):
            avIterates = torch.stack(allAvs[m])
            if k > 1: avIterates = torch.stack(allAvs[m][1:])
            avIterates = avIterates - torch.mean(avIterates, 0)
            allAvs[m] = avIterates

        return allWeights, allAvs, optimizer, net

    def getNet(self, params):
        i = 0
        model = deepcopy(self.clf).cuda()
        for p in model.parameters():
            L = len(p.flatten())
            param = params[i:(i + L)]
            p.data = param.view(p.size())
            i += L
        return model

    def fitBatchnorm(self, model):
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(), transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])
        model = model.cuda()
        for ind, (x, y, _) in enumerate(loader_tr):
            x, y = x.cuda(), y.cuda()
            output = model(x)
        return model

    def sampleNet(self, weights, iterates):
        nEns = len(weights)
        k = len(iterates[0])
        i = np.random.randint(nEns)
        z = torch.randn(k, 1)
        weightSample = weights[i].view(-1) - torch.mm(iterates[i].t(), z).view(-1) / np.sqrt(k)
        sampleNet = self.getNet(weightSample).cuda()
        sampleNet = self.fitBatchnorm(sampleNet)
        return sampleNet


    def predict(self, X, Y):
        if type(X) is np.ndarray:
            loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        else: 
            loader_te = DataLoader(self.handler(X.numpy(), Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        P = torch.zeros(len(Y)).long()
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                P[idxs] = pred.data.cpu()
        return P

    def predict_prob(self, X, Y, model=[], exp=True):
        if type(model) == list: model = self.clf

        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']), shuffle=False, **self.args['loader_te_args'])
        model = model.eval()
        probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = model(x)
                if exp: out = F.softmax(out, dim=1)
                probs[idxs] = out.cpu().data
        
        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += out.cpu().data
        probs /= n_drop
        
        return probs

    

    def get_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                embedding[idxs] = e1.data.cpu()
        
        return embedding

    

    

