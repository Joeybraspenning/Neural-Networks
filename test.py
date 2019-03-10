import numpy as np
from matplotlib import pyplot as plt


train_in = np.genfromtxt('./data/train_in.csv', delimiter=',')
train_out = np.genfromtxt('./data/train_out.csv', delimiter=',')
test_in = np.genfromtxt('./data/test_in.csv', delimiter=',')
test_out = np.genfromtxt('./data/test_out.csv', delimiter=',')


def sigmoid(x, a=1):
   return 1 / (1 + np.exp(-a*x))   



def MNIST_net(x, weights_1, weights_2):
   '''
   x = 257x1 input image vector, the last element is the bias
   OR 
   x = 257xN input image matrix, the last elements are biases
   weights_1 = 30x257 weights matrix
   weights_2 = 10x31 weights matrix
   '''
   a = sigmoid(weights_1.dot(x))
   try:
      b = sigmoid(weights_2.dot(np.append(a.T, np.ones(int(a.size/30)).reshape(-1,1), axis=1).T))
   except IndexError:
      b = sigmoid(weights_2.dot(np.append(a, 1)))

   return np.argmax(b, axis=0)



def mse(w1, w2):
   mse = np.mean((MNIST_net(image_matrix, w1, w2) - train_out)**2)
   return mse



def grdmse(weights_1, weights_2, temp1, temp2):
   eps = 0.1
   grad_1 = np.zeros(weights_1.size)
   grad_2 = np.zeros(weights_2.size)
   for i in range(weights_1.size + weights_2.size):
      if i < weights_1.size:
         grad_1[i] = (mse(weights_1 + eps*temp1[:,:,i], weights_2) - mse(weights_1, weights_2)) / eps

      elif i > weights_1.size:
         grad_2[i-weights_1.size] = (mse(weights_1, weights_2 + eps*temp2[:,:,i-weights_1.size]) - mse(weights_1, weights_2)) / eps

   return grad_1, grad_2


eta = 0.01

weights_1 = np.random.random((30, 257))
weights_2 = np.random.random((10, 31))
err = 0
tel=0

temp1 = np.zeros((weights_1.shape[0], weights_1.shape[1], weights_1.size))
temp2 = np.zeros((weights_2.shape[0], weights_2.shape[1], weights_2.size))
for i in range(weights_1.size + weights_2.size):
   if i < weights_1.size:
      idx = np.unravel_index(i, weights_1.shape)
      temp1[idx[0], idx[1], i] = 1

   elif i > weights_1.size:
      idx = np.unravel_index(i - weights_1.size, weights_2.shape)
      temp2[idx[0], idx[1], i-weights_1.size] = 1




image_matrix = np.zeros((256, len(train_in)))
for i in np.arange(len(train_in)):
   image_matrix[:,i] = train_in[i]
# Add a row on ones to the image matrix
image_matrix = np.vstack((image_matrix, np.ones(len(train_in))))

print(mse(weights_1, weights_2))
accuracy = 100*np.sum(train_out == MNIST_net(image_matrix, weights_1, weights_2))/len(train_out)

for i in range(5):
   grad1, grad2 = grdmse(weights_1, weights_2, temp1, temp2)
   weights_1 = weights_1 - eta*grad1.reshape(weights_1.shape)
   weights_2 = weights_2 - eta*grad2.reshape(weights_2.shape)

print(mse(weights_1, weights_2))
accuracy = 100*np.sum(train_out == MNIST_net(image_matrix, weights_1, weights_2))/len(train_out)
