import random
import time
import numpy as np
import plotly.graph_objs as go
from network import Network
from plotly.offline import plot
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def plot_2D_data(X,y):
    ti=np.where(y==1)[0]
    fi=np.where(y==0)[0]
    trace0 = go.Scatter(
        x = X[fi,0],
        y = X[fi,1],
        mode = 'markers',
        name = 'False',
        marker = dict(
            size = 4
        )
    )
    trace1 = go.Scatter(
        x = X[ti,0],
        y = X[ti,1],
        mode = 'markers',
        name = 'True',
        marker = dict(
            size = 4
        )
    )
    return [trace0,trace1]

def plot_classifier(trace,clf):
    x=np.linspace(-2,2,200)
    y=np.linspace(-2,2,200)
    x_t,y_t=np.meshgrid(x,y)
    x_t.ravel()
    z=clf.predict_proba(np.c_[x_t.ravel(),y_t.ravel()])[:,0]
    z=z.reshape(x_t.shape)
    trace.append(
        go.Heatmap(
            x=x,
            y=y,
            z=z,
            colorscale = 'Viridis',
            showscale=False
        )
    )
    layout = go.Layout(
        autosize= False,
        height = 800,
        width = 800
    )
    fig = go.Figure(
        data = trace,
        layout = layout
    )
    plot(
        fig,
        filename='test.html',
        image='png',
        image_filename='test',
        image_height = 800,
        image_width = 800,
        #auto_open = False,
        show_link = False
    )

def rand_2D_data(n):
    X=np.ndarray((n,2))
    y=np.ndarray((n),int)
    for i in range(n):
        X[i,:]=[random.uniform(-2,2),random.uniform(-2,2)]
        r=random.random()
        if(
            (X[i,0]-0.5)*(X[i,0]-0.5)+(X[i,1]-0.5)*(X[i,1]-0.5)<=0.4 or 
            (X[i,0]-0.5)*(X[i,0]-0.5)+(X[i,1]+0.5)*(X[i,1]+0.5)<=0.4 or 
            (X[i,0]+0.5)*(X[i,0]+0.5)+(X[i,1]-0.5)*(X[i,1]-0.5)<=0.4 or 
            (X[i,0]+0.5)*(X[i,0]+0.5)+(X[i,1]+0.5)*(X[i,1]+0.5)<=0.4
        ):
            if r<=0.95:
                y[i]=1
            else:
                y[i]=0
        else:
            if r<=0.95:
                y[i]=0
            else:
                y[i]=1
    return X,y

if __name__=='__main__':
    x,y = rand_2D_data(4000)
    trace = plot_2D_data(x,y)
    #clf = svm.SVC(probability=True)
    #clf=RandomForestClassifier(100)
    clf=Network((16,8,4), 'bgdm', epochs=20000)
    #clf=MLPClassifier((16,),solver='sgd',tol=1e-8,max_iter=10000,alpha=0)
    t=time.time()
    clf.fit(x,y)
    t=int((time.time()-t))
    print(t,"s")
    plot_classifier(trace,clf)
