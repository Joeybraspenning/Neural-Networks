import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import pickle

train_in = np.genfromtxt('./data/train_in.csv', delimiter=',')
train_out = np.genfromtxt('./data/train_out.csv', delimiter=',')
test_in = np.genfromtxt('./data/test_in.csv', delimiter=',')
test_out = np.genfromtxt('./data/test_out.csv', delimiter=',')

train_out = np.array(train_out, dtype='int')


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def sigmoid(x, a=1):
   return 1 / (1 + np.exp(-a*x))   

def relu(x):
   return np.maximum(0, x)



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








def run_MNSIT_net():
   weights_1 = 2*np.random.random((30, 257)) - 1
   weights_2 = 2*np.random.random((10, 31)) - 1

   image_matrix = np.zeros((256, len(train_in)))
   for i in np.arange(len(train_in)):
      image_matrix[:,i] = train_in[i]
   # Add a row on ones to the image matrix
   image_matrix = np.vstack((image_matrix, np.ones(len(train_in))))


    
   test_matrix = np.zeros((256, len(test_in)))
   for i in np.arange(len(test_in)):
      test_matrix[:,i] = test_in[i]
   # Add a row on ones to the image matrix
   test_matrix = np.vstack((test_matrix, np.ones(len(test_in))))



   for eta in [0.1, 1, 10]:
      tel = 0
      train_accuracy = defaultdict(list)
      test_accuracy = defaultdict(list)
      for j in range(20):
         weights_1 = 2*np.random.random((30, 257)) - 1
         weights_2 = 2*np.random.random((10, 31)) - 1
         accuracy = 100*np.sum(train_out == MNIST_net(image_matrix, weights_1, weights_2)[0])/len(train_out)

         tel = 0
         while accuracy != 100:
            tel +=1
            for i in range(len(train_out)):
               grad1, grad2 = mse_backprop(weights_1, weights_2, image_matrix[:, i], train_out[i])
               weights_1 -=  eta*grad1
               weights_2 -=  eta*grad2
         
            accuracy = 100*np.sum(train_out == MNIST_net(image_matrix, weights_1, weights_2)[0])/len(train_out)
            train_accuracy[j].append(accuracy)

            accuracy_test = 100*np.sum(test_out == MNIST_net(test_matrix, weights_1, weights_2)[0])/len(test_out)
            test_accuracy[j].append(accuracy_test)
            if tel > 50:
               break


         accuracy = 100*np.sum(train_out == MNIST_net(image_matrix, weights_1, weights_2)[0])/len(train_out)
         print('Accuracy = ', accuracy)

         accuracy_test = 100*np.sum(test_out == MNIST_net(test_matrix, weights_1, weights_2)[0])/len(test_out)
         print('Test Accuracy = ', accuracy_test)

      save_obj(train_accuracy, 'train_accuracy_sigmoid_{}'.format(eta))
      save_obj(test_accuracy, 'test_accuracy_sigmoid_{}'.format(eta))

def plot_MNIST_net():
   from matplotlib import rcParams
   rcParams['font.family'] = 'Latin Modern Roman'
   train_accuracy = defaultdict(lambda: defaultdict(list))
   test_accuracy = defaultdict(lambda: defaultdict(list))
   fig, ax = plt.subplots(3,2, sharex = True, sharey='row')
   for i, eta in enumerate([0.1, 1, 10]):
      train_accuracy[eta] = load_obj('train_accuracy_sigmoid_{}'.format(eta))
      test_accuracy[eta] = load_obj('test_accuracy_sigmoid_{}'.format(eta))
      mean_test_accuracy = []
      for j in range(20):
         ax[i, 0].plot(train_accuracy[eta][j], linewidth=0.5)
         ax[i, 1].plot(test_accuracy[eta][j], linewidth = 0.5)

         ax[i, 0].set_xticks([0, 10, 20, 30, 40, 50])
         ax[i, 1].set_xticks([0, 10, 20, 30, 40, 50])

         mean_test_accuracy.append(test_accuracy[eta][j][-1])
      print('Test_accuracy', eta, np.mean(mean_test_accuracy))



   ax[0, 0].text(0.05, 0.05, r'$\mathrm{{\eta}} = 0.01$', transform=ax[0, 0].transAxes, fontsize = 14, color = '#FF0000')
   ax[1, 0].text(0.05, 0.05, r'$\mathrm{{\eta}} = 0.1$', transform=ax[1, 0].transAxes, fontsize = 14, color = '#FF0000')
   ax[2, 0].text(0.05, 0.05, r'$\mathrm{{\eta}} = 1$', transform=ax[2, 0].transAxes, fontsize = 14, color = '#FF0000')


   fig.text(0.01, 0.5, 'Accuracy', va='center', rotation='vertical', fontsize = 20)
   fig.text(0.5, 0.01, 'Iterations', ha='center', fontsize = 20)
   ax[0, 0].set_title('Train')
   ax[0, 1].set_title('Test')
   plt.subplots_adjust(wspace = 0)
   plt.savefig('Task5_MNIST_progress.pdf')
   plt.show()

plot_MNIST_net()

