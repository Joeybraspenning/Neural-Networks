import numpy as np
from matplotlib import pyplot as plt


train_in = np.genfromtxt('./data/train_in.csv', delimiter=',')
train_out = np.genfromtxt('./data/train_out.csv', delimiter=',')
test_in = np.genfromtxt('./data/test_in.csv', delimiter=',')
test_out = np.genfromtxt('./data/test_out.csv', delimiter=',')

train_out = np.array(train_out, dtype='int')

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

   return (np.argmax(b, axis=0), b, a)


def mse_backprop(w1, w2, image_in, image_out):
   b_max, b, a = MNIST_net(image_in, w1, w2)

   try:
      b_temp = b.copy()
      b_temp[:, np.where(b_max == image_out)] = 0
      delta_2 = (1 - b_temp).dot(np.append(a.T, np.ones(int(a.size/30)).reshape(-1,1), axis=1))
      delta_1 = (a*(1-a) * (1-b_temp).T.dot(weights_2).T[:30,:]).dot(image_in.T)


   except IndexError:
      # delta_2 = (b_max != image_out) * (np.sum(b)*(1 - b)).reshape(-1, 1).dot(np.append(a, 1).reshape(-1, 1).T)


      delta_21 =  (((1 - b/np.sum(b)) * (1-b)).reshape(-1, 1)).dot(np.append(a, 1).reshape(-1, 1).T)
      delta_21[range(10) != image_out, :] = 0

      delta_22 = ((  (1 - b/np.sum(b)) * (b*(1-b))/(np.sum(b) - b) ).reshape(-1, 1)).dot(np.append(a, 1).reshape(-1, 1).T)
      delta_22[range(10) == image_out, :] = 0


      delta_2 = -delta_21 + delta_22

      temp1 = (((1 - b/np.sum(b)) * (1-b)).reshape(1, -1))
      temp1[:, range(10) != image_out] = 0
      delta_11 = ( temp1.dot(weights_2[:, :30]) * (a*(1-a)).reshape(1,-1) ).T.dot(image_in.reshape(1, -1))

      temp2 =((  (1 - b/np.sum(b)) * (b*(1-b))/(np.sum(b) - b) ).reshape(1, -1))
      temp2[:, range(10) == image_out] = 0
      delta_12 = ( temp2.dot(weights_2[:, :30]) * (a*(1-a)).reshape(1,-1) ).T.dot(image_in.reshape(1, -1))

      delta_1 = -delta_11 + delta_12


      # delta_1 = np.zeros_like(delta_1)


   return delta_1, delta_2




def mse(w1, w2, image_in, image_out):
   b_max, b, a = MNIST_net(image_in, w1, w2)
   # accuracy = 100*np.sum(image_out == b_max)/len(image_out)^M
   pos = np.where(b_max != image_out)[0]
   # print([[j, idx] for j, idx in zip(pos, image_out[pos])])
   # print(np.array([b[int(idx), j] for j, idx in zip(pos, image_out[pos])]))
   # print(np.array([b[int(idx2), k] for k, idx2 in zip(pos, np.argsort(b[:, pos], axis=0)[-2, :])]))
   ratio = np.array([b[int(idx), j] for j, idx in zip(pos, image_out[pos])]) / np.array([b[int(idx2), k] for k, idx2 in zip(pos, np.argsort(b[:, pos], axis=0)[-2, :])])
   return np.mean(ratio**2) + 10*((len(image_out) - len(pos))/len(image_out))**2




def mse_logcrossentropy(w1, w2, image_in, image_out):
   b_max, b, a= MNIST_net(image_in, w1, w2)

   log1 = np.sum(np.log( np.array([b[m,p] for p, m in enumerate(image_out)]) / np.sum(b, axis=0)))

   log2 = np.sum(np.log(1 -   b / np.sum(b, axis=0)  )) - np.sum(np.log(1 - np.array([b[m,p] for p, m in enumerate(image_out)]) / np.sum(b, axis=0)))

   return  -log1 + log2


def grdmse(weights_1, weights_2, temp1, temp2, image_in, image_out, eps1, eps2):
   eps = 0.01
   grad_1 = np.zeros(weights_1.shape)
   grad_2 = np.zeros(weights_2.shape)
   for i in range(weights_1.size + weights_2.size):
      if i < weights_1.size:
         grad_1[np.unravel_index(i, grad_1.shape)] = (mse_logcrossentropy(weights_1 + eps1[np.unravel_index(i, grad_1.shape)]*temp1[:,:,i], weights_2, image_in, image_out) - mse_logcrossentropy(weights_1, weights_2, image_in, image_out)) / eps

      elif i > weights_1.size:
         grad_2[np.unravel_index(i-weights_1.size, grad_2.shape)] = (mse_logcrossentropy(weights_1, weights_2 + eps2[np.unravel_index(i-weights_1.size, grad_2.shape)]*temp2[:,:,i-weights_1.size], image_in, image_out) - mse_logcrossentropy(weights_1, weights_2, image_in, image_out)) / eps

   return grad_1, grad_2



weights_1 = 2*np.random.random((30, 257)) - 1
weights_2 = 2*np.random.random((10, 31)) - 1

'''
temp1 = np.zeros((weights_1.shape[0], weights_1.shape[1], weights_1.size))
temp2 = np.zeros((weights_2.shape[0], weights_2.shape[1], weights_2.size))
for i in range(weights_1.size + weights_2.size):
   if i < weights_1.size:
      idx = np.unravel_index(i, weights_1.shape)
      temp1[idx[0], idx[1], i] = 1

   elif i > weights_1.size:
      idx = np.unravel_index(i - weights_1.size, weights_2.shape)
      temp2[idx[0], idx[1], i-weights_1.size] = 1
'''




image_matrix = np.zeros((256, len(train_in)))
for i in np.arange(len(train_in)):
   image_matrix[:,i] = train_in[i]
# Add a row on ones to the image matrix
image_matrix = np.vstack((image_matrix, np.ones(len(train_in))))

# print('MSE_old = ', mse(weights_1, weights_2, image_matrix, train_out))
# print('Cross entropy = ', mse_logcrossentropy(weights_1, weights_2, image_matrix, train_out))
# accuracy = 100*np.sum(train_out == MNIST_net(image_matrix, weights_1, weights_2)[0])/len(train_out)
# print('Accuracy = ', accuracy)
'''
image_input = np.array_split(image_matrix, 10, axis=1)
image_output = np.array_split(train_out, 10)


eta = 0.01
eps = 0.01
# for j in range(5):
#    image_input = np.array_split(image_matrix[:, np.random.permutation(image_matrix.shape[1])], 10, axis=1)
#    image_output = np.array_split(train_out[np.random.permutation(train_out.shape[0])], 10)
#    for i in range(10):
#       grad1, grad2 = grdmse(weights_1, weights_2, temp1, temp2, image_input[i], image_output[i])
#       weights_1 = weights_1 - eta*grad1
#       weights_2 = weights_2 - eta*grad2
# for j in range(10):
#    eps1 = np.abs(np.random.normal(eps, 5*eps, weights_1.shape))
#    eps2 = np.abs(np.random.normal(eps, 5*eps, weights_2.shape))
#    grad1, grad2 = grdmse(weights_1, weights_2, temp1, temp2, image_matrix, train_out, eps1, eps2)
#    weights_1 = weights_1 - eta*grad1*eps1/0.01
#    weights_2 = weights_2 - eta*grad2*eps2/0.01


#    print('MSE_old = ', mse(weights_1, weights_2, image_matrix, train_out))
#    print('Cross entropy = ', mse_logcrossentropy(weights_1, weights_2, image_matrix, train_out))
#    accuracy = 100*np.sum(train_out == MNIST_net(image_matrix, weights_1, weights_2)[0])/len(train_out)
#    print('Accuracy = ', accuracy)

'''
test_matrix = np.zeros((256, len(test_in)))
for i in np.arange(len(test_in)):
   test_matrix[:,i] = test_in[i]
# Add a row on ones to the image matrix
test_matrix = np.vstack((test_matrix, np.ones(len(test_in))))


# b_max, b, a = MNIST_net(image_matrix, weights_1, weights_2)
# print(np.max(b, axis=0), np.sort(b, axis=0)[-2, :])

eta = 1e-1
tel = 0
for j in range(10):
   weights_1 = 2*np.random.random((30, 257)) - 1
   weights_2 = 2*np.random.random((10, 31)) - 1
   accuracy = 100*np.sum(train_out == MNIST_net(image_matrix, weights_1, weights_2)[0])/len(train_out)

   while accuracy != 100:

      for i in range(len(train_out)):
         grad1, grad2 = mse_backprop(weights_1, weights_2, image_matrix[:, i], train_out[i])
         weights_1 -=  eta*grad1
         weights_2 -=  eta*grad2
   
      accuracy = 100*np.sum(train_out == MNIST_net(image_matrix, weights_1, weights_2)[0])/len(train_out)


   accuracy = 100*np.sum(train_out == MNIST_net(image_matrix, weights_1, weights_2)[0])/len(train_out)
   print('Accuracy = ', accuracy)

   accuracy_test = 100*np.sum(test_out == MNIST_net(test_matrix, weights_1, weights_2)[0])/len(test_out)
   print('Test Accuracy = ', accuracy_test)