#多项式生成使用指南
#poly = PolynomialFeatures(2)
#X1 = poly.fit_transform(X)
#scaler.fit(X1)
#X1 = scaler.transform(X1)
#print(X.shape,X1.shape)
import cmath
import numpy.random
import sklearn
import heapq
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import numpy as np

a = np.arange(10)
b = np.arange(10)
print(b[a<5])