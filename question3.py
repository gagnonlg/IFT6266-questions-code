import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from numpy.polynomial.polynomial import Polynomial


####################
# Config

# 1251
np.random.seed(715)
theano.config.floatX = 'float32'


####################
# Constants

# Fitted polynomial order
order = 2
# Number of data points to fit
ndata = 3
# Number of validation point
nvalid = 3
# The data generating polynomial
real_p = Polynomial([0.1, 1.2])
# Standard deviation of additive noise
noise_stddev = 0.5
# number of training epochs
nepochs = 15000


####################
# Data generation

def gen(ndata):
    trainX, trainY = real_p.linspace(n=ndata, domain=[-1,1])

    trainY += np.random.normal(0, noise_stddev, size=ndata)

    # transform the dataset in a suitable form for training
    trainX_ = np.zeros((ndata, order+1))
    for n in range(ndata):
        for i in range(order+1):
            trainX_[n, i] = trainX[n] ** i
    trainX = trainX_.astype('float32')
    trainY = trainY.astype('float32')

    return trainX, trainY

trainX, trainY = gen(ndata)
testX, testY = gen(nvalid)


####################
# Model definitions

# input and output
x = T.vector('x')
y = T.scalar('y')

# The early-stopped model
w = theano.shared(np.random.uniform(size=(order+1)).astype('float32'), name='w')
loss = T.pow(T.dot(w, x) - y, 2)
updates = [(w, w - 0.01 * T.grad(loss, w))]
train = theano.function([x, y], outputs=loss, updates=updates)
valid = theano.function([x, y], outputs=loss)

# The L2 regularized model
w2 = theano.shared(np.random.uniform(size=(order+1)).astype('float32'), name='w2')
loss2 = T.pow(T.dot(w2, x) - y, 2) + 0.5 * T.dot(w2, w2)
updates2 = [(w2, w2 - 0.01 * T.grad(loss2, w2))]
train2 = theano.function([x, y], outputs=loss2, updates=updates2)


#####################
# Training the models

# The L2 regularized model
for i in range(nepochs):
    for j in range(order+1):
        train2(trainX[j], trainY[j])

# The early stopped model
last = float('inf')
w_early = None

for i in range(15000):

    for j in range(ndata):
        train(trainX[j], trainY[j])

    vlosses = []
    for j in range(nvalid):
        vlosses.append(valid(testX[j], testY[j]))

    vloss = np.mean(vlosses)

    if last < vloss and w_early is None:
        w_early = np.array(w.get_value(), copy=True)
    else:
        last = vloss
        

#####################
# Graph

# training/validation data
plt.scatter(testX[:,1], testY, label='validation data')
plt.scatter(trainX[:,1], trainY, label='training data')


# early stopped fit
xt, yt = Polynomial(w_early).linspace(domain=[np.min(trainX[:,1]),np.max(trainX[:,1])])
plt.plot(xt, yt, label='early stop')

# L2 regularized fit
xt2, yt2 = Polynomial(w2.get_value()).linspace(domain=[np.min(trainX[:,1]),np.max(trainX[:,1])])
plt.plot(xt2, yt2, label='l2 reg')

# Unregularized fit
x0, y0 = Polynomial(w.get_value()).linspace(domain=[np.min(trainX[:,1]),np.max(trainX[:,1])])
plt.plot(x0, y0, label='no reg')

# The true function
xr,yr  = real_p.linspace(domain=[-1,1])
plt.plot(xr, yr, '--', label='true')
 
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('reg.png')
plt.show()

