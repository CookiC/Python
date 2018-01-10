"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a _feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

# Libraries
# Standard library
import time
import math
# Third-party libraries
import numpy as np
from numpy.random import randn


# Miscellaneous functions
def sigmoid(z):
    """Sigmoid function"""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_d(z):
    """The derivative of sigmoid function"""
    sz = sigmoid(z)
    return sz*(1-sz)

def relu(z):
    """Relu function"""
    return max(0,z)


def relu_d(z):
    """The dericative of relu function"""
    if z>0:
        return 1
    return 0

def tanh(z):
    ez = np.exp(z)
    enz = np.exp(-z)
    return (ez-enz)/(ez+enz)

def tanh_d(z):
    t = tanh(z)
    return 1-t*t

class Network(object):

    def __init__(self, sizes, solver='bgd', epochs=100):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.sizes = sizes
        if solver == 'sgd':
            pass
        elif solver == 'bgdm':
            self.solver = self.BGDM
        else:
            self.solver = self.BGD
        self.epochs = epochs
        self.alpha = 1
        self.weights = None
        self.biases = None
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train, x_test=None, y_test=None):
        s = set()
        for y in y_train:
            s.add(y)
        m = len(s)
        if m<3:
            m = 1
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.sizes = (x_train.shape[1],)+self.sizes+(m,)
        self.num_layers = len(self.sizes)
        self.biases = [randn(y, 1) for y in self.sizes[1:]]
        self.weights = [randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.solver()

    def predict_proba(self,x):
        n = len(x)
        y = np.zeros((n,2))
        y[:,1]=self._feedforward(x.T)
        y[:,0]=1-y[:,1]
        return y
    
    def predict(self,x):
        n = len(x)
        y = self._feedforward(x.T)
        return np.array(y>0.5, dtype='int')

    def _feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def BGD(self):
        """Train the neural network using batch gradient descent.``eta``
        is the learning rate."""
        h0 = self.cost()
        if self.y_test:
            n_test = len(self.y_test)
        for j in range(self.epochs):
            d_prob = np.abs(
                self._feedforward(self.x_train.T)-self.y_train
            ).reshape(-1)
            self.update_batch()
            h = self.cost()
            """if j%400==0:
                print("Cost is %.4f. Alpha is %.4f."%(h,self.alpha))
                if self.y_test:
                    print("Epoch {0}: {1} / {2}".format(
                        j, self.evaluate(self.x_test, self.y_test), n_test))
                else:
                    print("Epoch {0} complete".format(j))"""
    
    def BGDM(self):
        h0 = self.cost()
        x = self.x_train
        y = self.y_train
        n = len(y)
        intv = np.ones(n)
        clk = np.zeros(n)
        if self.y_test:
            n_test = len(self.y_test)
        beta = 0.05
        for j in range(self.epochs):
            idx = np.where(clk<=j)[0]
            m = len(idx)
            if m == 0:
                beta /= 2
                continue
            delta_b, delta_w = self.backprop_m(x[idx].T, y[idx])
            self.weights = [w-self.alpha/m*dw
                            for w, dw in zip(self.weights, delta_w)]
            self.biases = [b-self.alpha/m*db
                            for b, db in zip(self.biases, delta_b)]
            d_prob = np.abs(self._feedforward(x[idx].T)-y[idx]).reshape(-1)
            idx_p = idx[d_prob<beta]
            idx_n = idx[d_prob>=beta]
            intv[idx_p] *= 1.1
            intv[idx_n] = intv[idx_n]/1.1
            clk[idx] += intv[idx]

            h = self.cost()
            """if j%400==0:
                print("Cost is %.4f. Alpha is %.4f."%(h,self.alpha))
                if self.y_test:
                    print("Epoch {0}: {1} / {2}".format(
                        j, self.evaluate(self.x_test, self.y_test), n_test))
                else:
                    print("Epoch {0} complete".format(j))
        print(beta)"""
    
    def backprop_m(self, x, y):
        """ """
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        a_i = x
        a = [x]
        z = []
        for b, w in zip(self.biases, self.weights):
            z_i = np.dot(w, a_i)+b
            z.append(z_i)
            a_i = sigmoid(z_i)
            a.append(a_i)
        delta = self.cost_d(a[-1], y)*sigmoid_d(z[-1])
        delta_b[-1] = np.sum(delta,1,keepdims=True)
        delta_w[-1] = np.dot(delta, a[-2].T)
        for l in range(2, self.num_layers):
            sp = a[-l]*(1-a[-l])
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            delta_b[-l] = np.sum(delta,1,keepdims=True)
            delta_w[-l] = np.dot(delta, a[-l-1].T)
        return delta_b, delta_w

    def update_batch(self):
        x=self.x_train.T
        y=self.y_train
        n = len(y)
        delta_b, delta_w = self.backprop(x, y)
        l=0
        for dw in delta_w:
            l+=np.sum(dw*dw)
        for db in delta_b:
            l+=np.sum(db*db)
        l=math.sqrt(l)
        #self.gradient_checking(delta_w,delta_b)
        self.weights = [w-self.alpha/l*dw
                        for w, dw in zip(self.weights, delta_w)]
        self.biases = [b-self.alpha/l*db
                       for b, db in zip(self.biases, delta_b)]

    def backprop(self, x, y):
        """ """
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        # _feedforward
        a_i = x
        a = [x]  # list to store all the activations, layer by layer
        z = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z_i = np.dot(w, a_i)+b
            z.append(z_i)
            a_i = sigmoid(z_i)
            a.append(a_i)
        # backward pass
        delta = self.cost_d(a[-1], y)*sigmoid_d(z[-1])
        delta_b[-1] = np.sum(delta,1,keepdims=True)
        delta_w[-1] = np.dot(delta, a[-2].T)
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            sp = a[-l]*(1-a[-l])
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            delta_b[-l] = np.sum(delta,1,keepdims=True)
            delta_w[-l] = np.dot(delta, a[-l-1].T)
        return delta_b, delta_w

    def evaluate(self, x, y):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self._feedforward(x_i)), y_i)
                        for x_i, y_i in zip(x, y)]
        return sum(int(x_i == y_i) for x_i, y_i in test_results)
    
    def cost(self):
        y1=self._feedforward(self.x_train.T)
        y0=self.y_train
        return np.sum(-y0*np.log(y1)-(1-y0)*np.log(1-y1))

    def cost_d(self, y1, y0):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return -y0/y1+(1-y0)/(1-y1)
    
    def _gradient_checking(self,dw,db,esp=1e-6):
        dg = 0
        w = self.weights
        b = self.biases
        for i in range(len(w)):
            for j in range(w[i].shape[0]):
                for k in range(w[i][j].shape[0]):
                    w[i][j][k]+=esp
                    c1=self.cost()
                    w[i][j][k]-=2*esp
                    c2=self.cost()
                    w[i][j][k]+=esp
                    dw_c=(c1-c2)/(2*esp)
                    dg+=abs(dw_c-dw[i][j][k])
        for i in range(len(b)):
            for j in range(b[i].shape[0]):
                for k in range(b[i][j].shape[0]):
                    b[i][j][k]+=esp
                    c1=self.cost()
                    b[i][j][k]-=2*esp
                    c2=self.cost()
                    b[i][j][k]+=esp
                    db_c=(c1-c2)/(2*esp)
                    #print("db_c",db_c,"db",db[i][j][k])
                    dg+=abs(db_c-db[i][j][k])
        print("dg",dg)
