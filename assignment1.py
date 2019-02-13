import numpy as np
import sklearn.metrics

train_in = np.genfromtxt('./data/train_in.csv', delimiter=',')
train_out = np.genfromtxt('./data/train_out.csv', delimiter=',')
test_in = np.genfromtxt('./data/test_in.csv', delimiter=',')
test_out = np.genfromtxt('./data/test_out.csv', delimiter=',')

def Euclidean_distance(x1, x2, ax=0):
   return np.sqrt(np.sum((x1-x2)**2, axis = ax))

def Pairwise_distance(x1, x2, ax=0, measure='euclidean'):
   return sklearn.metrics.pairwise.paired_distances(x1, x2, metric=measure)

#Task 1
mean = np.empty((len(np.unique(train_out)), 256))
for i, num in enumerate(np.sort(np.unique(train_out))):
   mean[i, :] = np.mean(train_in[train_out == num], axis=0)

radius = np.empty(len(np.unique(train_out)))
for i, num in enumerate(np.unique(train_out)):
   radius[i] = np.max(Pairwise_distances(train_in[train_out == num], np.tile(mean[i, :], (train_in[train_out == num].shape[0], 1))))

number = np.empty(len(np.unique(train_out)))
for i, num in enumerate(np.unique(train_out)):
   number[i] = np.sum(train_out == num)

dist = np.empty((len(np.unique(train_out)), len(np.unique(train_out))))
for i in range(len(np.unique(train_out))):
   for j in range(len(np.unique(train_out))):
      dist[i,j] = Pairwise_distance(mean[i,:], mean[j,:])

#most difficult to seperate
temp = dist
temp[temp == 0] = 1e10
print(np.unravel_index(temp.argmin(), temp.shape))

#Task 2
classification_train = np.empty(train_in.shape[0])
for i in range(train_in.shape[0]):
   temp = []
   for j in np.sort(np.unique(train_out)):
      temp.append(Pairwise_distance(train_in[i, :], mean[int(j),:]))
   classification_train[i] = np.argmin(np.array(temp))

classification_test = np.empty(test_in.shape[0])
for i in range(test_in.shape[0]):
   temp = []
   for j in np.sort(np.unique(test_out)):
      temp.append(Pairwise_distance(test_in[i, :], mean[int(j),:]))
   classification_test[i] = np.argmin(np.array(temp))

#percentage of wrongly classified numbers
#print('Train: {:.2f}% wronly classified'.format(float(np.sum(train_out != classification_train))/len(train_out) * 100))
#print('Test: {:.2f}% wronly classified'.format(float(np.sum(test_out != classification_test))/len(test_out) * 100))
conf_mat_train = sklearn.metrics.confusion_matrix(train_out, classification_train)
conf_mat_test = sklearn.metrics.confusion_matrix(test_out, classification_test)


print(Euclidean_distance(train_in[0], train_in[1]))
print(train_in[0].shape)
print(Pairwise_distance(train_in[0:2], train_in[1:3]))


