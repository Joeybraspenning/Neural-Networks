import numpy as np
from sklearn import metrics
from collections import defaultdict
from matplotlib import pyplot as plt

train_in = np.genfromtxt('./data/train_in.csv', delimiter=',')
train_out = np.genfromtxt('./data/train_out.csv', delimiter=',')
test_in = np.genfromtxt('./data/test_in.csv', delimiter=',')
test_out = np.genfromtxt('./data/test_out.csv', delimiter=',')

def Euclidean_distance(x1, x2, ax=0):
   return np.sqrt(np.sum((x1-x2)**2, axis = ax))

def Pairwise_distance(x1, x2, ax=0, measure='euclidean'):
   return metrics.pairwise.paired_distances(x1, x2, metric=measure)

print(Pairwise_distance(train_in[0].reshape(1,-1), train_in[1].reshape(1,-1)))
#Task 1
mean = np.empty((len(np.unique(train_out)), 256))
for i, num in enumerate(np.sort(np.unique(train_out))):
   mean[i, :] = np.mean(train_in[train_out == num], axis=0)

radius = np.empty(len(np.unique(train_out)))
for i, num in enumerate(np.unique(train_out)):
   radius[i] = np.max(Pairwise_distance(train_in[train_out == num], np.tile(mean[i, :], (train_in[train_out == num].shape[0], 1))))

number = np.empty(len(np.unique(train_out)))
for i, num in enumerate(np.unique(train_out)):
   number[i] = np.sum(train_out == num)

dist = np.empty((len(np.unique(train_out)), len(np.unique(train_out))))
for i in range(len(np.unique(train_out))):
   for j in range(len(np.unique(train_out))):
      dist[i,j] = Pairwise_distance(mean[i,:].reshape(1,-1), mean[j,:].reshape(1,-1))

#most difficult to seperate
temp = dist
temp[temp == 0] = 1e10
print(np.unravel_index(temp.argmin(), temp.shape))

#Task 2
for method in ['cosine', 'manhattan', 'euclidean', 'l2', 'l1', 'cityblock']:
   print(method)
   classification_train = np.empty(train_in.shape[0])
   for i in range(train_in.shape[0]):
      temp = []
      for j in np.sort(np.unique(train_out)):
         temp.append(Pairwise_distance(train_in[i, :].reshape(1,-1), mean[int(j),:].reshape(1,-1), measure=method))
      classification_train[i] = np.argmin(np.array(temp))

   classification_test = np.empty(test_in.shape[0])
   for i in range(test_in.shape[0]):
      temp = []
      for j in np.sort(np.unique(test_out)):
         temp.append(Pairwise_distance(test_in[i, :].reshape(1,-1), mean[int(j),:].reshape(1,-1), measure=method))
      classification_test[i] = np.argmin(np.array(temp))

   #percentage of wrongly classified numbers
   print('Train: {:.2f}% wronly classified'.format(float(np.sum(train_out != classification_train))/len(train_out) * 100))
   print('Test: {:.2f}% wronly classified'.format(float(np.sum(test_out != classification_test))/len(test_out) * 100))
   conf_mat_train = metrics.confusion_matrix(train_out, classification_train)
   conf_mat_test = metrics.confusion_matrix(test_out, classification_test)



#Task 3
'''
Here we choose a feature to compare discriminate between two digits
The feature used is the number of pixels for which the pixel value > -1
'''
num_pixels = defaultdict(lambda: list())
for i in range(len(train_in)):
   if train_out[i] in [1,3]:
      num_pixels[int(train_out[i])].append(np.sum(train_in[i] > -1))

#Make histograms of the number of pixels for each category
plt.hist(num_pixels[1], color='red', histtype='step', bins=np.linspace(0,256,50))
plt.hist(num_pixels[3], color='blue', histtype='step', bins=np.linspace(0,256,50))
plt.xlabel('# pixels in image')
plt.ylabel('Counts')
plt.show()

#Calculate conditional probability P(X|C)
pdf_XC = defaultdict()
for key in num_pixels.keys():
   pdf_XC[key], binedges = np.histogram(num_pixels[key], bins=range(256), density=True)

#Make histogram of the pdfs
plt.hist(range(255), weights=pdf_XC[1], bins = np.linspace(0,256,50), histtype = 'step', color='red')
plt.hist(range(255), weights=pdf_XC[3], bins = np.linspace(0,256,50), histtype = 'step', color='blue')
plt.xlabel('# pixels in image')
plt.ylabel('pdf')
plt.show()

#calculate the probability for category C at every point P(C)
pdf_C = defaultdict()
pdf_C[1] = pdf_XC[1] / (pdf_XC[1] + pdf_XC[3])
pdf_C[3] = pdf_XC[3] / (pdf_XC[3] + pdf_XC[1])

pdf_C[1][np.isnan(pdf_C[1]) == True] = 0
pdf_C[3][np.isnan(pdf_C[3]) == True] = 0

#Calculate conditional probability P(C|X) at each point
pdf_CX = defaultdict()
for key in num_pixels.keys():
   pdf_CX[key] = pdf_XC[key] * pdf_C[key]

plt.plot(range(255), pdf_CX[1], color='red')
plt.plot(range(255), pdf_CX[3], color='blue')
plt.xlabel('# pixels in image')
plt.ylabel('P(C|X)')
plt.show()

'''
Now we treat the case where both are zero, classify as the one category which dominates closest to this point
'''
pdf_CX_temp = defaultdict(lambda: defaultdict())
for i in range(255):
   if (pdf_CX[3][i] == pdf_CX[1][i]) & (pdf_CX[3][i] == 0):
      d1 = np.min(np.abs(np.nonzero(pdf_CX[1])[0] - i)) #distance to 1
      d3 = np.min(np.abs(np.nonzero(pdf_CX[3])[0] - i)) #distance to 3
      if d1 < d3:
         pdf_CX_temp[1][i] = 1 #closer to 1
      if d1 > d3:
         pdf_CX_temp[3][i] = 1 #closer to 3

for key in num_pixels.keys():
   for i in pdf_CX_temp[key].keys():
      pdf_CX[key][i] = 1

#Plot the new P(C|X), note that the normalisation is irrelevant here
plt.plot(range(255), pdf_CX[1], color='red')
plt.plot(range(255), pdf_CX[3], color='blue')
plt.xlabel('# pixels in image')
plt.ylabel('P(C|X)')
plt.show()


'''
Now we try it on our test set.
First we select the part of the test set containing 1's and 3's
Then we calculate the number of pixels > -1 for each such image and classify according to whether P(1|num_pix) > or < P(3|num_pix)
Whenever P(1|num_pix) == P(3|num_pix) we randomly set it to each case with p = 0.5
'''

test_in_select = test_in[(test_out == 1) | (test_out == 3)]
test_out_select = test_out[(test_out == 1) | (test_out == 3)]
classification = np.empty(test_in_select.shape[0])
for i in range(len(test_in_select)):
   num_pix = np.sum(test_in_select[i] > -1)
   if pdf_CX[1][num_pix] > pdf_CX[3][num_pix]:
      classification[i] = 1
   elif pdf_CX[3][num_pix] > pdf_CX[1][num_pix]:
      classification[i] = 3
   elif (pdf_CX[3][num_pix] == pdf_CX[1][num_pix]) & (pdf_CX[3][num_pix] > 0): #They are equal
      if np.random.uniform() >= 0.5:
         classification[i] = 3 #Set to 3
      if np.random.uniform() < 0.5:
         classification[i] = 1 #Set to 1

print('Test: {:.2f}% wronly classified'.format(float(np.sum(test_out_select != classification))/len(test_out_select) * 100))
