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

def plot_digits():

    fig = plt.figure(figsize = (8,4))

    ax1 = fig.add_subplot(241)
    ax1.imshow(train_in[4].reshape(16,16), cmap='Greys')

    ax1.set_yticklabels([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_xticks([])

    #ax1.spines['top'].set_visible(True)
    #ax1.spines['right'].set_visible(True)

    ax2 = fig.add_subplot(242)
    ax2.imshow(train_in[36].reshape(16,16), cmap='Greys')

    ax2.set_yticklabels([])
    ax2.set_yticks([])
    ax2.set_xticklabels([])
    ax2.set_xticks([])

    ax3 = fig.add_subplot(243)
    ax3.imshow(train_in[9].reshape(16,16), cmap='Greys')

    ax3.set_yticklabels([])
    ax3.set_yticks([])
    ax3.set_xticklabels([])
    ax3.set_xticks([])

    ax4 = fig.add_subplot(244)
    ax4.imshow(train_in[8].reshape(16,16), cmap='Greys')

    ax4.set_yticklabels([])
    ax4.set_yticks([])
    ax4.set_xticklabels([])
    ax4.set_xticks([])

    ax5 = fig.add_subplot(245)
    ax5.imshow(train_in[17].reshape(16,16), cmap='Greys')

    ax5.set_yticklabels([])
    ax5.set_yticks([])
    ax5.set_xticklabels([])
    ax5.set_xticks([])

    ax6 = fig.add_subplot(246)
    ax6.imshow(train_in[52].reshape(16,16), cmap='Greys')

    ax6.set_yticklabels([])
    ax6.set_yticks([])
    ax6.set_xticklabels([])
    ax6.set_xticks([])

    ax7 = fig.add_subplot(247)
    ax7.imshow(train_in[93].reshape(16,16), cmap='Greys')

    ax7.set_yticklabels([])
    ax7.set_yticks([])
    ax7.set_xticklabels([])
    ax7.set_xticks([])

    ax8 = fig.add_subplot(248)
    ax8.imshow(train_in[678].reshape(16,16), cmap='Greys')

    ax8.set_yticklabels([])
    ax8.set_yticks([])
    ax8.set_xticklabels([])
    ax8.set_xticks([])

    #ax1.axis('off')
    fig.tight_layout()
    plt.savefig('mnist_images.pdf')

    plt.show()

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

    print(frequency)

    # Compute distances between the mean vectors
    dist = np.empty((N_digits, N_digits))
    for i in range(N_digits):
       for j in range(N_digits):
          dist[i,j] = Pairwise_distance(mean[i,:].reshape(1,-1), mean[j,:].reshape(1,-1))
          #print(Euclidean_distance(mean[i,:], mean[j,:]))
          #print(Pairwise_distance(mean[i,:].reshape(1,-1), mean[j,:].reshape(1,-1)))

    print(dist[7,9], dist[7,9])

    # Make figure
    xx, yy = np.meshgrid(np.linspace(-0.5, 9.5, 11),np.linspace(-0.5, 9.5, 11))

    plt.figure()
    plt.pcolor(xx,yy,dist,cmap='jet')
    plt.xticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'), size=12, fontweight='bold')
    plt.yticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'), size=12, fontweight='bold')
    plt.xlabel('Digit A', size=14)
    plt.ylabel('Digit B', size=14)

    for i in range(N_digits):
        for j in range(N_digits):

            distance = dist[i,j]
            distance = np.round(distance, decimals=1)

            if (i == j):
                pass
            elif (distance > 10):
                plt.annotate(distance, (j-0.45,i-0.2), color='k', fontsize=10, fontweight='bold')
            else:
                plt.annotate(distance, (j-0.3,i-0.2), color='k', fontsize=10, fontweight='bold')

    cbar = plt.colorbar(pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Distance $d(c_A, c_B)$ between cloud centers', size=13)
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
    #for method in ['minkowski']:

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

        # How many times does a digit occur in the training set?
        frequency_train = np.empty(N_digits)
        for i, num in enumerate(np.arange(N_digits)):
           frequency_train[i] = np.sum(train_out == num)

        # How many times does a digit occur in the test set?
        frequency_test = np.empty(N_digits)
        for i, num in enumerate(np.arange(N_digits)):
           frequency_test[i] = np.sum(test_out == num)


        freq_matrix_train = np.zeros((10,10))
        freq_matrix_test = np.zeros((10,10))
        for i in np.arange(10):
            freq_matrix_train[:, i] = frequency_train
            freq_matrix_test[:, i] = frequency_test

        # Make plots
        xx, yy = np.meshgrid(np.linspace(-0.5, 9.5, 11),np.linspace(-0.5, 9.5, 11))

        # Training set
        plt.figure()
        plt.pcolor(xx,yy,conf_mat_train/freq_matrix_train*100,cmap='jet',vmax=12)
        plt.xlabel('Digit B', size=14)
        plt.ylabel('Digit A', size=14)
        plt.xticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'), size=12, fontweight='bold')
        plt.yticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'), size=12, fontweight='bold')
        plt.title('Training Set', size=14)
        cbar = plt.colorbar(pad=0.02)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('Classifications of A as B (%)', size=13)
        cbar.set_ticks([0, 2, 4, 6, 8, 10, 12])
        plt.gca().set_aspect('equal', adjustable='box')
        for i in range(N_digits):
            for j in range(N_digits):

                percentage = conf_mat_train[i,j]/frequency_train[i]*100

                if (percentage == 100):
                    percentage = int(percentage)
                    plt.annotate(percentage, (j-0.4,i-0.2), color='w', fontsize=10, fontweight='bold')
                elif (percentage == 0):
                    percentage = int(percentage)
                    plt.annotate(percentage, (j-0.1,i-0.2), color='w', fontsize=10, fontweight='bold')
                elif(percentage > 10):
                    percentage = np.round(percentage, decimals=1)
                    plt.annotate(percentage, (j-0.45,i-0.2), color='w', fontsize=10, fontweight='bold')
                else:
                    percentage = np.round(percentage, decimals=1)
                    plt.annotate(percentage, (j-0.35,i-0.2), color='w', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig('task2_'+method+'_train_per.pdf')
        #plt.show()

        # Test set
        plt.figure()
        plt.pcolor(xx,yy,conf_mat_test/freq_matrix_test*100,cmap='jet',vmax=12)
        plt.xlabel('Digit B', size=14)
        plt.ylabel('Digit A', size=14)
        plt.xticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'), size=12, fontweight='bold')
        plt.yticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'), size=12, fontweight='bold')
        plt.title('Test Set', size=14)
        cbar = plt.colorbar(pad=0.02)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('Classifications of A as B (%)', size=13)
        cbar.set_ticks([0, 2, 4, 6, 8, 10, 12])
        plt.gca().set_aspect('equal', adjustable='box')
        for i in range(N_digits):
            for j in range(N_digits):

                percentage = conf_mat_test[i,j]/frequency_test[i]*100

                if (percentage == 100):
                    percentage = int(percentage)
                    plt.annotate(percentage, (j-0.4,i-0.2), color='w', fontsize=10, fontweight='bold')
                elif (percentage == 0):
                    percentage = int(percentage)
                    plt.annotate(percentage, (j-0.1,i-0.2), color='w', fontsize=10, fontweight='bold')
                elif(percentage > 10):
                    percentage = np.round(percentage, decimals=1)
                    plt.annotate(percentage, (j-0.45,i-0.2), color='w', fontsize=10, fontweight='bold')
                else:
                    percentage = np.round(percentage, decimals=1)
                    plt.annotate(percentage, (j-0.35,i-0.2), color='w', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig('task2_'+method+'_test_per.pdf')
        #plt.show()


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
           plt.hist(num_pixels[digits[0]], color='red', histtype='bar', alpha=0.5, bins=np.linspace(1,256,256), label=str(digits[0]))
           plt.hist(num_pixels[digits[1]], color='blue', histtype='bar', alpha=0.5, bins=np.linspace(1,256,256), label=str(digits[1]))
           plt.xticks(size=12)
           plt.yticks(size=12)
           plt.xlabel('Number of non-white pixels in image', size=14)
           plt.ylabel('Number of occurrences', size=14)
           plt.xlim([20, 220])
           plt.legend(frameon=False, prop=dict(size=16))
           plt.tight_layout()
           #plt.savefig('task3_histogram.pdf')
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
    plt.xlabel('Digit A', size=14)
    plt.ylabel('Digit B', size=14)

    if (data_set == 'test'):
        plt.title('Test Set', fontsize=14)
    else:
        plt.title('Training Set', fontsize=14)

    plt.xticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'), size=12, fontweight='bold')
    plt.yticks(np.arange(10), ('0','1','2','3','4','5','6','7','8','9'), size=12, fontweight='bold')
    cbar = plt.colorbar(pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Accuracy (%)', size=13)
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

    # Add a row of ones to the image matrix
    image_matrix = np.vstack((image_matrix, np.ones(len(train_in))))

    accuracy_list = np.zeros(N_iter)

    ############################################################################
    print('')
    print('TRAINING NETWORK...')
    print('')

    for iter in np.arange(N_iter):

        accuracy, misclassified_items = compute_accuracy(weights_matrix, image_matrix, train_out)

        # For each iteration store accuracy in list
        accuracy_list[iter] = accuracy

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

    return weights_matrix, accuracy_list


def task_4():

    weights_matrix, accuracy_list = train_network()

    plt.figure()
    plt.plot(accuracy_list)
    plt.xlabel('# Iterations', size=12)
    plt.ylabel('Accuracy', size=12)
    plt.show()

    image_matrix = np.zeros((256, len(test_in)))
    for i in np.arange(len(test_in)):
        image_matrix[:,i] = test_in[i]

    # Add a row on ones to the image matrix
    image_matrix = np.vstack((image_matrix, np.ones(len(test_in))))

    accuracy, misclassified_items = compute_accuracy(weights_matrix, image_matrix, test_out)

    print('')
    print('APPLYING NETWORK TO TEST DATA...')
    print('')

    print('Accuracy (%): ' + str(accuracy) + \
          ' -- # Misclassified items: ' + str(len(misclassified_items)))

#plot_digits()
#task_4()
task_1()
#task_2()
#task_3(data_set='test')
#task_3(show_plots=False, data_set='test')
