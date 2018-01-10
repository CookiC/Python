import random
import numpy as np
from sklearn.neighbors import NearestNeighbors

class Smote:
    def __init__(self,samples,N=10,k=5,seed=0):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
        self.seed=seed

    def OverSampling(self):
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        random.seed(self.seed)
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            self._populate(N,i,nnarray)
        return self.synthetic


    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1

def SMOTE(X,y,seed=0):
    Pi = []
    Ni = []
    P = 0
    N = 0
    for i in range(len(y)):
        if y[i]==b'T':
            Pi.append(i)
            P += 1
        else:
            Ni.append(i)
            N += 1
    s=Smote(np.array(X[Pi,:]),(N-P)/P*100,seed=seed)
    T=s.OverSampling()
    X=np.vstack((X,T))
    y=np.hstack((y,np.array([b'T']*T.shape[0])))
    return X,y

def Sample(a,n):
    return a[np.random.randint(len(a), size=int(n))]

def RandomSample(X,y,rate=1):
    size = len(y)
    I = np.arange(size)
    I = Sample(I,size*rate)
    return X[I,:],y[I]

#P is the minority class
def UnderSample(x,y,rate=1):
    Pi = []
    Ni = []
    n = len(y)
    for i in range(n):
        if y[i]==1:
            Pi.append(i)
        else:
            Ni.append(i)
    N = int(n*rate+1.5)//2
    Ni = Sample(np.array(Ni),N)
    Pi = Sample(np.array(Pi),N)
    Si = np.concatenate((Ni, Pi))
    return x[Si,:],y[Si]

#P is the minority class
def OverSample(X,y,rate=1):
    Pi = []
    Ni = []
    P = 0
    N = 0
    for i in range(len(y)):
        if y[i]==b'T':
            Pi.append(i)
            P += 1
        else:
            Ni.append(i)
            N += 1
    Ni = Sample(Ni,N*rate)
    Pi = Sample(Pi,N*rate)
    Si = Pi+Ni
    return X[Si,:],y[Si]

def CountTF(pred,test):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(test)):
        if(test[i]==1):
            if(pred[i]==1):
                TP+=1
            else:
                FP+=1
        else:
            if(pred[i]==1):
                FN+=1
            else:
                TN+=1
    return TP,FP,TN,FN

def MCC(pred,test):
    TP, FP, TN, FN = CountTF(pred, test)
    if(TP+FP==0 or TP+FN==0 or TN+FP==0 or TN+FN==0):
        return 0
    return (TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5

if __name__=='__main__':
    pass