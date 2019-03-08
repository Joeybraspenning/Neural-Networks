import numpy as np
from matplotlib import pyplot as plt


train_in = np.genfromtxt('./data/train_in.csv', delimiter=',')
train_out = np.genfromtxt('./data/train_out.csv', delimiter=',')
test_in = np.genfromtxt('./data/test_in.csv', delimiter=',')
test_out = np.genfromtxt('./data/test_out.csv', delimiter=',')


def sigmoid(x, a=1):
   return 1 / (1 + np.exp(-a*x))   



def MNIST_net(x, weights):
   a = np.zeros((30, weights.shape[1]))
   b = np.zeros((10, weights.shape[1]))

   '''
   x = 256x1 input image vector
   weights = 8049x1 input weights vector
   '''
   for i in range(30):
      a[i] = sigmoid(x.dot(weights[256*i:256*i+256]) + weights[7680+i]) #All 30 inputs + bias



   for j in range(10):
      b[j] = sigmoid(np.diag(a.T.dot(weights[7710 + 30*j:7740 + 30*j])) + weights[8040+j])
   return np.argmax(b, axis=0)



def mse(weights):
   se = 0
   for i, x in enumerate(train_in):
      se += (MNIST_net(x, weights.reshape(-1,1)) - train_out[i])**2
   mse = np.mean(np.sqrt(se))
   return mse

def mse_one(weights, num):
   se = (MNIST_net(train_in[num], weights) - train_out[num])**2
   return np.sqrt(se)


def grdmse(weights, num):
   eps = 0.001
   temp = np.repeat(weights.reshape(-1,1), 8050, axis=1) + eps*np.diag(np.ones(8050))

   grad = (mse_one(temp, num) - mse_one(weights.reshape(-1,1), num))/eps

  
   # for i in range(8050):
   #    grad[i] = (mse_one(temp[i], num) - mse_one(weights, num))/ eps


   return grad


eta = 0.01

weights = np.random.random(8050)
err = 0
tel=0

print(mse(weights))

for k in range(5):
   for num in range(len(train_in)):
      print(num)
      weights = weights - eta*grdmse(weights, num)
      
   print(mse_one2(weights))



'''
while err != 0:
   tel +=1
   msqe.append(mse(weights))
   weights = weights - eta * grdmse(weights)
   err = 0
   for x in itertools.product(range(2), repeat=2):
      err += (int(np.round(xor_net(x[0], x[1], weights))) - np.logical_xor(x[0], x[1]))**2
   mis.append(err)
   if tel > 1e6:
      break

'''