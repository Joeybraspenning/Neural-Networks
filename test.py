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

   return (np.argmax(b, axis=0), b)


def mse(w1, w2, image_in, image_out):
   b_max, b = MNIST_net(image_in, w1, w2)
   # accuracy = 100*np.sum(image_out == b_max)/len(image_out)^M
   pos = np.where(b_max != image_out)[0]
   # print([[j, idx] for j, idx in zip(pos, image_out[pos])])
   # print(np.array([b[int(idx), j] for j, idx in zip(pos, image_out[pos])]))
   # print(np.array([b[int(idx2), k] for k, idx2 in zip(pos, np.argsort(b[:, pos], axis=0)[-2, :])]))
   ratio = np.array([b[int(idx), j] for j, idx in zip(pos, image_out[pos])]) / np.array([b[int(idx2), k] for k, idx2 in zip(pos, np.argsort(b[:, pos], axis=0)[-2, :])])
   return np.mean(ratio**2) + 10*((len(image_out) - len(pos))/len(image_out))**2




def mse_logcrossentropy(w1, w2, image_in, image_out):
   b_max, b = MNIST_net(image_in, w1, w2)

   log1 = np.sum(np.log( np.array([b[m,p] for p, m in enumerate(image_out)]) / np.sum(b, axis=0)))

   log2 = np.sum(np.log(1 -   b / np.sum(b, axis=0)  )) - np.sum(np.log(1 - np.array([b[m,p] for p, m in enumerate(image_out)]) / np.sum(b, axis=0)))

   return -log1 + log2


def grdmse(weights_1, weights_2, temp1, temp2, image_in, image_out):
   eps = 0.001
   grad_1 = np.zeros(weights_1.shape)
   grad_2 = np.zeros(weights_2.shape)
   for i in range(weights_1.size + weights_2.size):
      if i < weights_1.size:
         grad_1[np.unravel_index(i, grad_1.shape)] = (mse_logcrossentropy(weights_1 + eps*temp1[:,:,i], weights_2, image_in, image_out) - mse_logcrossentropy(weights_1, weights_2, image_in, image_out)) / eps

      elif i > weights_1.size:
         grad_2[np.unravel_index(i-weights_1.size, grad_2.shape)] = (mse_logcrossentropy(weights_1, weights_2 + eps*temp2[:,:,i-weights_1.size], image_in, image_out) - mse_logcrossentropy(weights_1, weights_2, image_in, image_out)) / eps

   return grad_1, grad_2


eta = 0.1

weights_1 = 2*np.random.random((30, 257)) - 1
weights_2 = 2*np.random.random((10, 31)) - 1


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

print('MSE_old = ', mse(weights_1, weights_2, image_matrix, train_out))
print('Cross entropy = ', mse_logcrossentropy(weights_1, weights_2, image_matrix, train_out))
accuracy = 100*np.sum(train_out == MNIST_net(image_matrix, weights_1, weights_2)[0])/len(train_out)
print('Accuracy = ', accuracy)

image_input = np.array_split(image_matrix, 10, axis=1)
image_output = np.array_split(train_out, 10)

# for j in range(5):
#    image_input = np.array_split(image_matrix[:, np.random.permutation(image_matrix.shape[1])], 10, axis=1)
#    image_output = np.array_split(train_out[np.random.permutation(train_out.shape[0])], 10)
#    for i in range(10):
#       grad1, grad2 = grdmse(weights_1, weights_2, temp1, temp2, image_input[i], image_output[i])
#       weights_1 = weights_1 - eta*grad1
#       weights_2 = weights_2 - eta*grad2
for j in range(10):
   grad1, grad2 = grdmse(weights_1, weights_2, temp1, temp2, image_matrix, train_out)
   weights_1 = weights_1 - eta*grad1
   weights_2 = weights_2 - eta*grad2


   print('MSE_old = ', mse(weights_1, weights_2, image_matrix, train_out))
   print('Cross entropy = ', mse_logcrossentropy(weights_1, weights_2, image_matrix, train_out))
   accuracy = 100*np.sum(train_out == MNIST_net(image_matrix, weights_1, weights_2)[0])/len(train_out)
   print('Accuracy = ', accuracy)