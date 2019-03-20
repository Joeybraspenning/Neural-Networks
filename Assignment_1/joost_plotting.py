import numpy as np
from sklearn import metrics
from collections import defaultdict
from matplotlib import pyplot as plt
import itertools

train_in = np.genfromtxt('./data/train_in.csv', delimiter=',')
train_out = np.genfromtxt('./data/train_out.csv', delimiter=',')
test_in = np.genfromtxt('./data/test_in.csv', delimiter=',')
test_out = np.genfromtxt('./data/test_out.csv', delimiter=',')

N_digits = 10
N_pixels = 256

def Euclidean_distance(x1, x2, ax=0):
   return np.sqrt(np.sum((x1-x2)**2, axis = ax))

def Pairwise_distance(x1, x2, ax=0, measure='euclidean'):
   return metrics.pairwise.paired_distances(x1, x2, metric=measure)

def task_1():

    # Compute mean position vector for every digit
    mean = np.empty((N_digits, N_pixels))
    for i, num in enumerate(np.arange(N_digits)):
       mean[i, :] = np.mean(train_in[train_out == num], axis=0)

    # Compute radius (= maximum distance) for every digit
    radius = np.empty(N_digits)
    for i, num in enumerate(np.arange(N_digits)):
       radius[i] = np.max(Pairwise_distance(train_in[train_out == num], np.tile(mean[i, :], (train_in[train_out == num].shape[0], 1))))

    # Count how many times a digit is present in the training set
    frequency = np.empty(N_digits)
    for i, num in enumerate(np.arange(N_digits)):
       frequency[i] = np.sum(train_out == num)

    # Compute distances between the mean vectors
    dist = np.empty((N_digits, N_digits))
    for i in range(N_digits):
       for j in range(N_digits):
          dist[i,j] = Pairwise_distance(mean[i,:].reshape(1,-1), mean[j,:].reshape(1,-1))
          #print(Euclidean_distance(mean[i,:], mean[j,:]))
          #print(Pairwise_distance(mean[i,:].reshape(1,-1), mean[j,:].reshape(1,-1)))

    # Make figure
    xx, yy = np.meshgrid(np.linspace(-0.5, 9.5, 11),np.linspace(-0.5, 9.5, 11))

    plt.figure()
    plt.pcolor(xx,yy,dist,cmap='jet')
    plt.xticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'))
    plt.yticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'))
    plt.title('Distances between mean positions of digits')
    cbar = plt.colorbar()
    cbar.set_label('distance')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # Find the pair of digits whose mean vectors lie closest together
    temp = dist
    temp[temp == 0] = 1e10
    print(np.unravel_index(temp.argmin(), temp.shape))


def task_2():

    # Compute mean position vector for every digit
    mean = np.empty((N_digits, N_pixels))
    for i, num in enumerate(np.arange(N_digits)):
       mean[i, :] = np.mean(train_in[train_out == num], axis=0)

    for method in ['cosine', 'manhattan', 'euclidean', 'l2', 'l1', 'cityblock']:

        classification_train = np.empty(train_in.shape[0])
        classification_test = np.empty(test_in.shape[0])

        # Classify training data
        for i in range(train_in.shape[0]):
           temp_distances = []
           for j in range(N_digits):
              temp_distances.append(Pairwise_distance(train_in[i, :].reshape(1,-1), mean[int(j),:].reshape(1,-1), measure=method))
           classification_train[i] = np.argmin(np.array(temp_distances))

        # Clasify test data
        for i in range(test_in.shape[0]):
            temp_distances = []
            for j in range(N_digits):
                temp_distances.append(Pairwise_distance(test_in[i, :].reshape(1,-1), mean[int(j),:].reshape(1,-1), measure=method))
            classification_test[i] = np.argmin(np.array(temp_distances))

        # Print diagnostics and compute confusion matrices
        print('Method: ' + method)
        print('Train: {:.2f}% wronly classified'.format(float(np.sum(train_out != classification_train))/len(train_out) * 100))
        print('Test: {:.2f}% wronly classified'.format(float(np.sum(test_out != classification_test))/len(test_out) * 100))
        conf_mat_train = metrics.confusion_matrix(train_out, classification_train)
        conf_mat_test = metrics.confusion_matrix(test_out, classification_test)

        # Make plots
        xx, yy = np.meshgrid(np.linspace(-0.5, 9.5, 11),np.linspace(-0.5, 9.5, 11))

        # Training set
        plt.figure()
        plt.pcolor(xx,yy,conf_mat_train,cmap='jet',vmax=10)
        plt.xticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'))
        plt.yticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'))
        plt.title('Training set: confusion matrix ('+method+')')
        cbar = plt.colorbar()
        cbar.set_label('occurences')
        plt.gca().set_aspect('equal', adjustable='box')
        for i in range(N_digits):
            for j in range(N_digits):
                plt.annotate(conf_mat_train[i,j], (j-0.2,i-0.2), color='w', fontsize=10, fontweight='bold')
        plt.show()

        # Test set
        plt.figure()
        plt.pcolor(xx,yy,conf_mat_test,cmap='jet',vmax=10)
        plt.xticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'))
        plt.yticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'))
        plt.title('Test set: confusion matrix ('+method+')')
        cbar = plt.colorbar()
        cbar.set_label('occurences')
        plt.gca().set_aspect('equal', adjustable='box')
        for i in range(N_digits):
            for j in range(N_digits):
                plt.annotate(conf_mat_test[i,j], (j-0.2,i-0.2), color='w', fontsize=10, fontweight='bold')
        plt.show()


def task_3():

    # Feature: number of non-negative pixels

    quality = np.zeros((10,10))
    for digits in itertools.combinations(range(10), 2):
       num_pixels = defaultdict(lambda: list())
       for i in range(len(train_in)):
          if train_out[i] in digits:
             num_pixels[int(train_out[i])].append(np.sum(train_in[i] > -1))

       # Make histograms of the number of pixels for each category
       plt.figure()
       plt.hist(num_pixels[digits[0]], color='red', histtype='step', bins=np.linspace(0,256,50), label=str(digits[0]))
       plt.hist(num_pixels[digits[1]], color='blue', histtype='step', bins=np.linspace(0,256,50), label=str(digits[1]))
       plt.xlabel('# non-negative pixels in image')
       plt.ylabel('Counts')
       plt.legend()
       plt.show()

       # BAYES: P(digit|pixels) ~ P(pixels|digit)P(digit)

       # Calculate conditional probability P(X|C) = P(pixels|digit)
       pdf_XC = defaultdict()
       for key in num_pixels.keys():
          pdf_XC[key], binedges = np.histogram(num_pixels[key], bins=range(256), density=True)

       # Make histogram of the pdfs
       plt.figure()
       plt.hist(range(255), weights=pdf_XC[digits[0]], bins = np.linspace(0,256,50), histtype = 'step', color='green', label=str(digits[0]))
       plt.hist(range(255), weights=pdf_XC[digits[1]], bins = np.linspace(0,256,50), histtype = 'step', color='cyan', label=str(digits[1]))
       plt.xlabel('# non-negative pixels in image')
       plt.ylabel('PDF')
       plt.legend()
       plt.show()

       # calculate the probability for category C at every point P(C)
       pdf_C = defaultdict()
       pdf_C[digits[0]] = pdf_XC[digits[0]] / (pdf_XC[digits[0]] + pdf_XC[digits[1]])
       pdf_C[digits[1]] = pdf_XC[digits[1]] / (pdf_XC[digits[0]] + pdf_XC[digits[1]])

       pdf_C[digits[0]][np.isnan(pdf_C[digits[0]]) == True] = 0
       pdf_C[digits[1]][np.isnan(pdf_C[digits[1]]) == True] = 0

       #Calculate conditional probability P(C|X) at each point
       pdf_CX = defaultdict()
       for key in num_pixels.keys():
           pdf_CX[key] = pdf_XC[key] * pdf_C[key]

       plt.plot(range(255), pdf_CX[digits[0]], color='red')
       plt.plot(range(255), pdf_CX[digits[1]], color='blue')
       plt.xlabel('# pixels in image')
       plt.ylabel('P(C|X)')
       plt.show()







task_2()

"""

   #plt.plot(range(255), pdf_CX[1], color='red')
   #plt.plot(range(255), pdf_CX[3], color='blue')
   #plt.xlabel('# pixels in image')
   #plt.ylabel('P(C|X)')
   #plt.show()


   pdf_CX_temp = defaultdict(lambda: defaultdict())
   for i in range(255):
      if (pdf_CX[digits[0]][i] == pdf_CX[digits[1]][i]) & (pdf_CX[digits[0]][i] == 0):
         d1 = np.min(np.abs(np.nonzero(pdf_CX[digits[0]])[0] - i)) #distance to 1
         d2 = np.min(np.abs(np.nonzero(pdf_CX[digits[1]])[0] - i)) #distance to 3
         if d1 < d2:
            pdf_CX_temp[digits[0]][i] = 1 #closer to 1
         if d1 > d2:
            pdf_CX_temp[digits[1]][i] = 1 #closer to 3

   for key in num_pixels.keys():
      for i in pdf_CX_temp[key].keys():
         pdf_CX[key][i] = 1

   #Plot the new P(C|X), note that the normalisation is irrelevant here
   #plt.plot(range(255), pdf_CX[1], color='red')
   #plt.plot(range(255), pdf_CX[3], color='blue')
   #plt.xlabel('# pixels in image')
   #plt.ylabel('P(C|X)')
   #plt.show()

'''
   #Now we try it on our test set.
   #First we select the part of the test set containing 1's and 3's
   #Then we calculate the number of pixels > -1 for each such image and classify according to whether P(1|num_pix) > or < P(3|num_pix)
   # Whenever P(1|num_pix) == P(3|num_pix) we randomly set it to each case with p = 0.5
'''
   test_in_select = test_in[(test_out == digits[0]) | (test_out == digits[1])]
   test_out_select = test_out[(test_out == digits[0]) | (test_out == digits[1])]
   classification = np.empty(test_in_select.shape[0])
   for i in range(len(test_in_select)):
      num_pix = np.sum(test_in_select[i] > -1)
      if pdf_CX[digits[0]][num_pix] > pdf_CX[digits[1]][num_pix]:
         classification[i] = digits[0]
      elif pdf_CX[digits[1]][num_pix] > pdf_CX[digits[0]][num_pix]:
         classification[i] = digits[1]
      elif (pdf_CX[digits[0]][num_pix] == pdf_CX[digits[1]][num_pix]) & (pdf_CX[digits[0]][num_pix] > 0): #They are equal
         if np.random.uniform() >= 0.5:
            classification[i] = digits[0] #Set to first number
         if np.random.uniform() < 0.5:
            classification[i] = digits[1] #Set to second number

   quality[digits[0], digits[1]] = float(np.sum(test_out_select != classification))/len(test_out_select)
   quality[digits[1], digits[0]] = float(np.sum(test_out_select != classification))/len(test_out_select)

   print('Test {}, {}: {:.2f}% wrongly classified'.format(digits[0], digits[1], float(np.sum(test_out_select != classification))/len(test_out_select) * 100))

plt.pcolor(quality)
plt.show()
'''

#Task 4

def quality_measure(data_in, data_out, w):
   d = -np.ones((10,len(data_in)))
   for i in range(len(data_in)):
      d[int(data_out[i]), i] = 1

   y = w.T.dot(data_in.T)
   idx = y*d > 0

   percentage = float(np.sum(idx))/np.size(y) * 100.

   correct = np.sum(((y*d)[d>0] > 0))/float(len(data_in))
   false_positives = np.sum((y[d<0] > 0)) /(9*float(len(data_in)))

   return percentage, correct, false_positives


w = np.random.rand(256,10)

print("train:", quality_measure(train_in, train_out, w))
print("test:", quality_measure(test_in, test_out, w))
print("\\\\\\\\\\\\\\\\\\\\")
eta = 0.01
for j in range(10):
   num = 0
   for i, im in enumerate(train_in):
      im = np.reshape(im, (256,1)).T # add 1 as 257th element

      d = -np.ones(10)
      d[int(train_out[i])] = 1

      y = im.dot(w)[0]

      idx = np.where(y*d < 0)[0]
      num += len(idx)
      w[:, idx] = w[:, idx] + eta * d[idx] * im.T
   print("train:", quality_measure(train_in, train_out, w), num)

   print("test:", quality_measure(test_in, test_out, w))
   print("\\\\\\\\\\\\\\\\\\\\")
"""
