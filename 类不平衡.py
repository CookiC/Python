import cmath
import ensemble
import random
import sklearn
import time
import numpy as np
from copy import copy
from copy import deepcopy
from io import StringIO
from numpy import ndarray
from network import Network
from sampling import SMOTE
from sampling import UnderSample
from sampling import MCC
from scipy.io import arff
from sklearn import svm
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def CountPN(y):
    P = 0
    N = 0
    for e in y:
        if e == b'T':
            P += 1
        else:
            N += 1
    return P,N

class Evaluation:
    def __init__(self):
        self.recs={}
    
    def clear(self):
        self.recs.clear()
    
    def run(self):
        self.clf.fit(self.x_train, self.y_train)
        prob = self.clf.predict_proba(self.x_test)
        y_prob = prob[:,1]
        #print(yProb)
        auc = roc_auc_score(self.y_test, y_prob)
        mcc = MCC(np.array(y_prob>0.5,dtype='int'), self.y_test)
        if isinstance(self.clf,ensemble.Bagging):
            k=self.clf.k_statistic()
            auc_train=self.clf.auc_train()
        else:
            k=0
            auc_train=0
        if mcc > 1:
            print(mcc)
            time.sleep(100)
        if not(self.clf.name in self.recs):
            self.recs[self.clf.name]=[]
        self.recs[self.clf.name].append((auc,auc_train,mcc,k))
        #print("%7.3f%7.3f%7.3f%7.3f"%(auc,auc_train,mcc,k))
    
    def average(self):
        for i in self.recs.keys():
            rec=self.recs[i]
            n = len(rec)
            m = len(rec[0])
            avg=np.zeros(m+2)
            for e in rec:
                for j in range(m):
                    avg[j]+=e[j]
            for j in range(m):
                avg[j]/=n
            for e in rec:
                avg[m]+=(e[0]-avg[0])*(e[0]-avg[0])
                avg[m+1]+=(e[2]-avg[2])*(e[2]-avg[2])
            self.recs[i]=tuple(avg)

    def print(self):
        for e in self.recs.items():
            print("%17s"%e[0],end='')
            for i in range(len(e[1])-2):
                print("%12.3f"%e[1][i],end='')
            #for f in e[1]:
            #    print("%10.3f"%f,end='')
            print()

class SVM(svm.SVC):
    def __init__(self):
        super(SVM,self).__init__(probability=True,random_state=7)

def LoadArff(path):
    data,meta = arff.loadarff(path)
    instNum=data.size
    attrNum=len(data[0])
    x = ndarray(shape=(instNum,attrNum-1))
    for i in range(instNum):
        for j in range(attrNum-1):
            x[i][j] = data[i][j]
    y = ndarray(shape=(instNum),dtype=int)
    for i in range(instNum):
        if(data[i][-1]==b'Y' or data[i][-1]==b'1'):
            y[i]=1
        else:
            y[i]=0
    scaler=MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    return x,y

def ClassifiersSet(seed=77):
    n = 10
    meths=[]
    #clf=DecisionTreeClassifier()
    #clf.name="DecisionTree"
    #meths.append(clf)
    clf=RandomForestClassifier(n)
    #clf.name="RandomForest"
    #meths.append(clf)
    clf=ensemble.UnderBagging(DecisionTreeClassifier(), n, seed=seed)
    clf.name="UnderBagging"
    meths.append(clf)
    clf=ensemble.RandomForest(n, seed=seed)
    clf.name="RandomForest"
    meths.append(clf)
    clf=ensemble.UnderRandomForest(n, seed=seed)
    clf.name="UnderRandomForest"
    meths.append(clf)
    """clf=ensemble.Bagging(
        MLPClassifier((8,4,2),solver='adam',max_iter=2000,tol=1e-8,
            learning_rate_init=0.01),
        n,
        seed=seed
    )
    clf.name="MLP(8,4,2)+Bag"
    meths.append(clf)
    clf=ensemble.Bagging3(
        MLPClassifier((8,4,2),solver='adam',max_iter=2000,tol=1e-8,
            learning_rate_init=0.01),
        n,
        seed=seed
    )
    clf.name="MLP(8,4,2)+Bag3"
    meths.append(clf)"""
    clf=ensemble.Bagging(
        Network((8,4,4), 'bgdm', 2000),
        n,
        seed=seed
    )
    clf.name="MLP(8,4,2)+Bag"
    meths.append(clf)
    clf=ensemble.Bagging3(
        Network((8,4,4), 'bgdm', 2000),
        n,
        seed=seed
    )
    clf.name="MLP(8,4,2)+Bag3"
    meths.append(clf)
    return meths

if __name__=='__main__':
    #按大小排序
    #dataName = ("velocity-1.6","synapse-1.2","jedit-4.0","ivy-2.0","MW1","CM1",
    #    "ant-1.7","xalan-2.6","camel-1.6","EQ","PC1","PC3","LC","PC4","JDT",
    #    "PDE","ML")
    #不平衡比例在四倍以上
    #dataName = ("camel-1.6","PDE","PC4","ML","CM1","PC3","ivy-2.0","MW1","LC",
    #"PC1")
    #不平衡比例在四倍以下
    #dataName = ('EQ','JDT','ant-1.7','jedit-4.0','synapse-1.2','velocity-1.6',
    # 'xalan-2.6')
    #NASA datasets
    dataName = ("CM1","MW1","PC1","PC3","PC4")
    clfs = ClassifiersSet()
    t = time.time()
    for i in range(len(dataName)):
        path = 'E:\\scientific research\\experiment\\data\\EXP\\'+dataName[i]+'.arff'
        x, y = LoadArff(path)
        eval = Evaluation()
        kfold = sklearn.model_selection.KFold(10)
        print("-----",dataName[i],"-----")
        j = 0
        for i_train, i_test in kfold.split(y):
            j += 1
            #print(">Round",j)
            eval.x_train = x[i_train]
            eval.x_test = x[i_test]
            eval.y_train = y[i_train]
            eval.y_test = y[i_test]
            for clf in clfs:
                #print(">>",clf.name)
                eval.clf=deepcopy(clf)
                eval.run()
        print("%17s%12s%12s%12s%12s"%("","AUC","AUC_train","MCC","k-statistic"))
        #,"AUC variance","MCC variance"))
        eval.average()
        eval.print()
        eval.clear()
    print(int(time.time()-t),'s')
    #time.sleep(10000)