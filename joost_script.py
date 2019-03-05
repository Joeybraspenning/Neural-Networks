import numpy as np
from sklearn import metrics
from collections import defaultdict
from matplotlib import pyplot as plt
import itertools
import sys

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
    plt.xticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'), size=12, fontweight='bold')
    plt.yticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'), size=12, fontweight='bold')
    plt.xlabel('Digit A', size=12)
    plt.ylabel('Digit B', size=12)
    cbar = plt.colorbar(pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Distance $d(c_A, c_B)$ between cloud centers', size=12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig('task1_distances.pdf')

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
        plt.pcolor(xx,yy,conf_mat_train,cmap='jet',vmax=20)
        plt.xlabel('Digit A', size=12)
        plt.ylabel('Digit B', size=12)
        plt.xticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'), size=12, fontweight='bold')
        plt.yticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'), size=12, fontweight='bold')
        plt.title('Training set $-$ Method: '+method)
        cbar = plt.colorbar(pad=0.02)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('# Classifications of A as B', size=12)
        plt.gca().set_aspect('equal', adjustable='box')
        for i in range(N_digits):
            for j in range(N_digits):
                if conf_mat_train[i,j]/100.0 > 1:
                    plt.annotate(conf_mat_train[i,j], (j-0.4,i-0.2), color='w', fontsize=10, fontweight='bold')
                else:
                    plt.annotate(conf_mat_train[i,j], (j-0.2,i-0.2), color='w', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig('task2_'+method+'_train.pdf')
        plt.show()

        # Test set
        plt.figure()
        plt.pcolor(xx,yy,conf_mat_test,cmap='jet',vmax=20)
        plt.xlabel('Digit A', size=12)
        plt.ylabel('Digit B', size=12)
        plt.xticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'), size=12, fontweight='bold')
        plt.yticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'), size=12, fontweight='bold')
        plt.title('Test set $-$ Method: '+method)
        cbar = plt.colorbar(pad=0.02)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('# Classifications of A as B', size=12)
        plt.gca().set_aspect('equal', adjustable='box')
        for i in range(N_digits):
            for j in range(N_digits):
                if conf_mat_train[i,j]/100.0 > 1:
                    plt.annotate(conf_mat_test[i,j], (j-0.4,i-0.2), color='w', fontsize=10, fontweight='bold')
                else:
                    plt.annotate(conf_mat_test[i,j], (j-0.2,i-0.2), color='w', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig('task2_'+method+'_test.pdf')
        plt.show()


def task_3(show_plots=False, data_set='test'):

    # Feature: number of non-negative pixels
    quality = np.full((10,10), np.nan)
    for digits in itertools.combinations(range(10), 2):

    ############################# Loop over all combiations of digits ################################

       num_pixels = defaultdict(lambda: list())
       for i in range(len(train_in)):
          if train_out[i] in digits:
             num_pixels[int(train_out[i])].append(np.sum(train_in[i] > -1)) # dictionary that contains a list of pixel numbers for each digit

       if show_plots:
           # Histograms for number of pixels in each category
           plt.figure()
           plt.hist(num_pixels[digits[0]], color='red', histtype='bar', alpha=0.5, bins=np.linspace(0,256,50), label=str(digits[0]))
           plt.hist(num_pixels[digits[1]], color='blue', histtype='bar', alpha=0.5, bins=np.linspace(0,256,50), label=str(digits[1]))
           plt.xticks(size=12)
           plt.yticks(size=12)
           plt.xlabel('# Non-negative pixels in image', size=12)
           plt.ylabel('# Counts', size=12)
           plt.legend(frameon=False, prop=dict(size=16))
           plt.tight_layout()
           plt.show()

       # BAYES: P(digit|pixels) ~ P(pixels|digit)P(digit)

       # Calculate conditional probability P(X|C) = P(pixels|digit)
       pdf_XC = defaultdict()
       for key in num_pixels.keys():
          pdf_XC[key], binedges = np.histogram(num_pixels[key], bins=range(256), density=True)

       if show_plots:
           # Histogram of likelihood: P(X|C) = P(pixels|digit)
           plt.figure()
           plt.hist(range(255), weights=pdf_XC[digits[0]], bins = np.linspace(0,256,50), histtype = 'bar', alpha=0.5, color='gold', label=str(digits[0]))
           plt.hist(range(255), weights=pdf_XC[digits[1]], bins = np.linspace(0,256,50), histtype = 'bar', alpha=0.5, color='cyan', label=str(digits[1]))
           plt.xticks(size=12)
           plt.yticks(size=12)
           plt.xlabel('# Non-negative pixels in image', size=12)
           plt.ylabel('P(pixels | digit)', size=12)
           plt.legend(frameon=False, prop=dict(size=16))
           plt.tight_layout()
           plt.show()

       # calculate the prior probability P(digit)
       # P(A) = A/(A+B) and P(B) = B/(A+B), such that P(A) + P(B) = 1
       pdf_C = defaultdict()
       pdf_C[digits[0]] = len(num_pixels[digits[0]])/(len(num_pixels[digits[0]]) + len(num_pixels[digits[1]]))
       pdf_C[digits[1]] = len(num_pixels[digits[1]])/(len(num_pixels[digits[0]]) + len(num_pixels[digits[1]]))

       # Calculate the posterior probability P(digit|pixels) = P(pixels|digit)*P(digit)
       pdf_CX = defaultdict()
       for key in num_pixels.keys():
           pdf_CX[key] = pdf_XC[key]*pdf_C[key]

       if show_plots:
           plt.figure()
           plt.hist(range(255), weights=pdf_CX[digits[0]], bins = np.linspace(0,256,256), histtype = 'bar', alpha=0.5, color='gold', label=str(digits[0]))
           plt.hist(range(255), weights=pdf_CX[digits[1]], bins = np.linspace(0,256,256), histtype = 'bar', alpha=0.5, color='cyan', label=str(digits[1]))
           plt.xticks(size=12)
           plt.yticks(size=12)
           plt.xlabel('# Non-negative pixels in image', size=12)
           plt.ylabel('P(digit | pixels) = P(pixels | digit) P(digit)', size=12)
           plt.legend(frameon=False, prop=dict(size=16))
           plt.tight_layout()
           plt.show()

       '''
       Now we try it on our test set.
       First we select the part of the test set containing 1's and 3's
       Then we calculate the number of pixels > -1 for each such image and classify according to whether P(1|num_pix) > or < P(3|num_pix)
       Whenever P(1|num_pix) == P(3|num_pix) we randomly set it to each case with p = 0.5
       '''

       if (data_set == 'test'):
           test_in_select = test_in[(test_out == digits[0]) | (test_out == digits[1])]
           test_out_select = test_out[(test_out == digits[0]) | (test_out == digits[1])]
       elif (data_set == 'train'):
          test_in_select = train_in[(train_out == digits[0]) | (train_out == digits[1])]
          test_out_select = train_out[(train_out == digits[0]) | (train_out == digits[1])]
       else:
           raise ValueError('chose either the training or the test set')

       classification = np.empty(test_in_select.shape[0])
       for i in range(len(test_in_select)):
           num_pix = np.sum(test_in_select[i] > -1)
           if pdf_CX[digits[0]][num_pix] > pdf_CX[digits[1]][num_pix]:
               classification[i] = digits[0]
           elif pdf_CX[digits[1]][num_pix] > pdf_CX[digits[0]][num_pix]:
               classification[i] = digits[1]
         #elif (pdf_CX[digits[0]][num_pix] == pdf_CX[digits[1]][num_pix]) & (pdf_CX[digits[0]][num_pix] > 0): #They are equal
           else:
               if np.random.uniform() >= 0.5:
                   classification[i] = digits[0] #Set to first number
               else:
                   classification[i] = digits[1] #Set to second number

       quality[digits[0], digits[1]] = float(np.sum(test_out_select == classification))/len(test_out_select)
       quality[digits[1], digits[0]] = float(np.sum(test_out_select == classification))/len(test_out_select)

       print(data_set + ' {}, {}: {:.2f}% correctly classified'.format(digits[0], digits[1], float(np.sum(test_out_select != classification))/len(test_out_select) * 100))

    # Make plots
    xx, yy = np.meshgrid(np.linspace(-0.5, 9.5, 11),np.linspace(-0.5, 9.5, 11))

    print(quality)

    plt.figure()
    plt.pcolor(xx,yy,100*quality,cmap='jet') #,vmax=100)
    plt.xlabel('Digit A', size=12)
    plt.ylabel('Digit B', size=12)

    if (data_set == 'test'):
        plt.title('Test Set', fontsize=12)
    else:
        plt.title('Training Set', fontsize=12)

    plt.xticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'), size=12, fontweight='bold')
    plt.yticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'), size=12, fontweight='bold')
    cbar = plt.colorbar(pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Accuracy (%)', size=12)
    plt.gca().set_aspect('equal', adjustable='box')
    for i in range(N_digits):
        for j in range(N_digits):
            if (i != j):
                plt.annotate(int(np.round(100*quality[i,j])), (j-0.2,i-0.2), color='w', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('task3_'+data_set+'.pdf')
    plt.show()


def compute_accuracy(w_matrix, x_matrix, true_numbers):

    # Find the index of the maximum activation for each image
    output_numbers = np.argmax(w_matrix.dot(x_matrix), axis=0)

    # Compute the accuracy (%)
    accuracy = 100*np.sum(true_numbers == output_numbers)/len(true_numbers)

    # Indices of misclassified items
    misclassified_items = np.where(true_numbers != output_numbers)[0]

    return accuracy, misclassified_items


def train_network():

    N_perceptrons = 10

    # Learning rate
    eta = 0.01
    # Number of iterations over training set
    N_iter = 100

    # Draw weights from uniform distribution between -1 and 1
    weights_matrix = 2*np.random.random((N_perceptrons, N_pixels+1)) - 1

    # Store the whole training set in a matrix
    image_matrix = np.zeros((256, len(train_in)))
    for i in np.arange(len(train_in)):
        image_matrix[:,i] = train_in[i]

    # Add a row on ones to the image matrix
    image_matrix = np.vstack((image_matrix, np.ones(len(train_in))))

    ############################################################################
    print('')
    print('TRAINING NETWORK...')
    print('')

    for iter in np.arange(N_iter):

        accuracy, misclassified_items = compute_accuracy(weights_matrix, image_matrix, train_out)
        print('Iteration: ' + str(iter) + ' -- Accuracy (%): ' + str(accuracy) + \
              ' -- # Misclassified items: ' + str(len(misclassified_items)))

        for i in np.arange(len(train_in)):

            true_number = int(train_out[i])
            activation = weights_matrix.dot(image_matrix[:,i])

            # Find digits that give a higher activation than the true digit
            mask = np.where(activation > activation[true_number])[0]

            if len(mask) > 0:
                weights_matrix[mask, :] = weights_matrix[mask, :] - eta*image_matrix[:,i]
                weights_matrix[true_number, :] = weights_matrix[true_number, :] + eta*image_matrix[:,i]

    return weights_matrix

def task_4():

    weights_matrix = train_network()

    image_matrix = np.zeros((256, len(test_in)))
    for i in np.arange(len(test_in)):
        image_matrix[:,i] = test_in[i]

    # Add a row on ones to the image matrix
    image_matrix = np.vstack((image_matrix, np.ones(len(test_in))))

    accuracy, misclassified_items = compute_accuracy(weights_matrix, image_matrix, test_out)

    print('')
    print('APPLYING NETWORK TO TEST DATA...')
    print('')

    print('Iteration: ' + str(iter) + ' -- Accuracy (%): ' + str(accuracy) + \
          ' -- # Misclassified items: ' + str(len(misclassified_items)))


task_4()
#task_1()
#task_2()
#task_3(data_set='test')
#task_3(data_set='train')
