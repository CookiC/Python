import math
import random
import sampling
import time
import numpy as np
from copy import deepcopy
from copy import copy
from sampling import MCC
from sampling import Sample
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

class Bagging:
    def __init__(self,clf,num_e=20,seed=0):
        self.num_e=num_e
        self.clf=clf
        self.clfs = []
        self.k=0
        self.auc=0
        random.seed(seed)
    
    def _calculate_k(self,A,B):
        m=len(A)
        a=0
        b=0
        c=0
        d=0
        for i in range(m):
            if A[i]==1:
                if B[i]==1:
                    a+=1
                else:
                    b+=1
            else:
                if B[i]==1:
                    c+=1
                else:
                    d+=1
        p1 = (a+d)/m
        p2 = ((a+b)*(a+c)+(c+d)*(b+d))/(m*m)
        if p2==1:
            return 0
        return (p1-p2)/(1-p2)

    def _record_k(self,pred):
        sum_k = 0
        cnt_k = 0
        for i in range(self.num_e):
            for j in range(i+1,self.num_e):
                sum_k += self._calculate_k(pred[:,i],pred[:,j])
                cnt_k += 1
        if cnt_k > 0:
            self.k=sum_k/cnt_k
        else:
            self.k=1
    
    def _vote(self,x):
        n = len(x)
        pred = np.ndarray(shape=(n,self.num_e),dtype=int)
        prob = np.ndarray(shape=(n,self.num_e),dtype=float)
        self.prob = np.zeros(shape=(n,2),dtype=float)
        for i in range(self.num_e):
            prob[:,i] = self.clfs[i].predict_proba(x)[:,1]
            pred[:,i] = prob[:,i]>0.5
            self.prob[:,1] += prob[:,i]
        self.prob[:,1] /= self.num_e
        self.prob[:,0] = 1-self.prob[:,1]
        self._record_k(pred)

    def _train(self):
        for i in range(self.num_e):
            clf=deepcopy(self.clf)
            xi,yi=sampling.RandomSample(self.x,self.y,1)
            clf.fit(xi,yi)
            self.clfs.append(clf)
    
    def fit(self,x,y):
        self.x=x
        self.y=y
        self._train()

    def predict(self,x):
        self._vote(x)
        return np.array(self.prob[:,0]<=self.prob[:,1],dtype=int)
    
    def predict_proba(self,x):
        self._vote(x)
        return np.array(self.prob)

    def k_statistic(self):
        return self.k

    def auc_train(self):
        sum_auc = 0
        for clf in self.clfs:
            y_prob = clf.predict_proba(self.x)[:,1]
            y_true = np.array(self.y)
            sum_auc += roc_auc_score(y_true,y_prob)
        return sum_auc/self.num_e

class RandomForest(Bagging):
    def __init__(self,num_e=20,seed=0):
        super(RandomForest, self).__init__(DecisionTreeClassifier(),num_e, seed)
    
    def _train(self):
        attr_num = self.x.shape[1]
        m = int(attr_num/4+0.5)
        self.clf.set_params(max_features=m)
        for i in range(self.num_e):
            clf=deepcopy(self.clf)
            xi,yi=sampling.RandomSample(self.x,self.y)
            clf.fit(xi,yi)
            self.clfs.append(clf)

class UnderRandomForest(RandomForest):
    def _train(self):
        attr_num = self.x.shape[1]
        m = int(attr_num/4+0.5)
        self.clf.set_params(max_features=m)
        for i in range(self.num_e):
            clf=deepcopy(self.clf)
            xi,yi=sampling.UnderSample(self.x, self.y)
            clf.fit(xi,yi)
            self.clfs.append(clf)

class UnderBagging(Bagging):
    def _train(self):
        for i in range(self.num_e):
            clf = deepcopy(self.clf)
            xi,yi = sampling.UnderSample(self.x, self.y)
            clf.fit(xi, yi)
            self.clfs.append(clf)

class UnderBagging3(Bagging):
    def UnderSample(self,X,y):
        n = len(y)
        Pi = []
        Ni = []
        P = 0
        N = 0
        for i in range(n):
            if y[i]==b'T':
                Pi.append(i)
                P += 1
            else:
                Ni.append(i)
                N += 1
        rateN = random.uniform(0.2,0.8)
        rateP = random.uniform(0.2,0.8)
        Ni = random.sample(Ni,int(N*rateN))
        Pi = random.sample(Pi,int(P*rateP))
        train_i = Pi+Ni
        vis=np.ones(n,bool)
        for i in train_i:
            vis[i] = 0
        valid_i = []
        for i in range(n):
            if vis[i]:
                valid_i.append(i)
        return X[train_i,:],y[train_i],X[valid_i,:],y[valid_i]

    def fit(self,X,y):
        m = 4
        mccs = np.ndarray(self.n*m,dtype=float)
        clfs = np.ndarray(self.n*m,dtype=object)
        for i in range(self.n*m):
            clf=deepcopy(self.clf)
            X_train,y_train,X_valid,y_valid = self.UnderSample(X,y)
            clf.fit(X_train,y_train)
            prod = clf.predict_proba(X_valid)
            mcc=MCC(np.array(y_valid==b'T',dtype=float),prod[:,1])
            clfs[i] = clf
            mccs[i] = mcc
        idx=np.argsort(-mccs)[0:self.n]
        self.clfs=clfs[idx]
        mccs=mccs[idx]
        #print("n:",self.n,"MCC:",sum(mccs)/self.n)

class RandomBagging(Bagging):
    
    def _vote(self,x):
        n = len(x)
        pred = np.ndarray(shape=(n,self.num_e),dtype=int)
        prob = np.ndarray(shape=(n,self.num_e),dtype=float)
        self.prob = np.zeros(shape=(n,2),dtype=float)
        for i in range(self.num_e):
            prob[:,i] = self.clfs[i].predict_proba(x[:,self.attr_clf[i]])[:,1]
            pred[:,i] = prob[:,i]>0.5
            self.prob[:,1] += prob[:,i]
        self.prob[:,1] /= self.num_e
        self.prob[:,0] = 1-self.prob[:,1]
        self._record_k(pred)
    
    def _train(self):
        attr_num = self.x.shape[1]
        m = int(math.sqrt(attr_num)+0.5)
        attr_idx = np.arange(attr_num)
        self.attr_clf = np.ndarray((self.num_e,m), int)
        for i in range(self.num_e):
            clf = deepcopy(self.clf)
            xi,yi = sampling.RandomSample(self.x, self.y, 2)
            np.random.shuffle(attr_idx)
            self.attr_clf[i] = attr_idx[:m]
            clf.fit(xi[:,self.attr_clf[i]], yi)
            self.clfs.append(clf)
    
    def auc_train(self):
        sum_auc = 0
        for i in range(self.num_e):
            y_prob = self.clfs[i].predict_proba(self.x[:,self.attr_clf[i]])[:,1]
            sum_auc += roc_auc_score(self.y,y_prob)
        return sum_auc/self.num_e

class Bagging2(Bagging):
    def _RandomSample(self,X,y,rate=1):
        size = len(y)
        U = np.arange(size)
        train = Sample(U,size*rate)
        valid = np.setdiff1d(U,train)
        return X[train,:],y[train],X[valid,:],y[valid]
    
    def fit(self,X,y):
        cut1=0
        cut2=0
        r = len(y)
        pred = np.ndarray(shape=(r,self.n),dtype=int)
        sum_mcc = 0
        cut_mcc = 0.1
        cut_k = 0.55
        for i in range(self.n):
            flag = True
            while flag:
                clf=deepcopy(self.clf)
                X_train,y_train,X_valid,y_valid = self._RandomSample(X,y,0.7)
                clf.fit(X_train,y_train)
                pred[:,i] = np.array(clf.predict(X)==b'T',dtype=int)
                flag=False
                mcc = MCC(clf.predict(X_valid),y_valid)
                if mcc < cut_mcc:
                    flag = True
                    cut1 += 1
                if not flag:
                    for j in range(i):
                        if self._cal_k(pred[:,i],pred[:,j])>cut_k:
                            flag=True
                            cut2+=1
                            break
                #if cut1*8 < cut2:
                #    cut_mcc+=0.01
                #    cut_k+=0.01
                #else:
                #    cut_mcc-=0.01
                #    cut_k-=0.01
                #if cut1+cut2 > i*4:
                #    cut_mcc-=0.005
                #    cut_k+=0.005
                #else:
                #    cut_mcc+=0.005
                #    cut_k-=0.005
            sum_mcc += mcc
            self.clfs.append(clf)
        print("%.2f"%(sum_mcc/self.n),cut1,cut2)

class Bagging3(Bagging):
    def _train(self):
        for i in range(self.num_e):
            clf=deepcopy(self.clf)
            clf.fit(self.x,self.y)
            self.clfs.append(clf)