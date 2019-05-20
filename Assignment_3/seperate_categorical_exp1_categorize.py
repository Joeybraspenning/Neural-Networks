import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Flatten, MaxPooling2D, TimeDistributed
from keras.datasets import mnist
from keras.layers.normalization import BatchNormalization
import numpy as np
import tensorflow as tf
import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name +'.pkl', 'rb') as f:
        return pickle.load(f)


# spectra = load_obj('./bachelor_data/spectra_matrix_exp1.pickle')
# categories = load_obj('./bachelor_data/input_matrix_exp1.pickle')[:, :, 0]


# print(np.unique(categories[1,:]))

# print(spectra.shape)
# print(categories.shape)

# np.save('spectra_exp1', spectra)
# np.save('categories_exp1', categories)

spectra = np.load('spectra_exp1.npy')
categories = np.load('categories_exp1.npy')


# spectra -= np.expand_dims(np.min(spectra, axis=1), axis=1)
# print(np.sum(np.isnan(spectra)))
# print(np.sum(np.max(spectra, axis=1) == 0))
# print(np.where(np.max(spectra, axis=1) == 0))
# print(spectra[np.max(spectra, axis=1) == 0][0])
# spectra = spectra[np.max(spectra, axis=1) != 0]
# spectra = spectra.T/np.max(spectra, axis=1)


# print(categories.shape)

# print(np.max(spectra, axis=1).shape)
median = np.median(spectra, axis=1)
spectra = ((spectra.T - median) / np.max(spectra, axis=1)) + median
spectra = spectra.T

# print(spectra.shape)
# print(np.unique(categories[:,1]))

idx = np.random.permutation(spectra.shape[0])
train_idx = idx[:int(0.9*len(idx))]
test_idx = idx[int(0.9*len(idx)):]
spectra_train = np.expand_dims(spectra[train_idx, :], axis=2)
spectra_test = np.expand_dims(spectra[test_idx, :], axis=2)
categories_train = np.array(categories[train_idx, :], dtype='int')
categories_test = np.array(categories[test_idx, :], dtype='int')
# categories_train = categories[train_idx, :]
# categories_test = categories[test_idx, :]

categorical_train = np.zeros((categories_train.shape[0], 7, 2))
for i in range(len(categories_train)):
  for j, string in enumerate(categories_train[i,:]):
    if string:
      categorical_train[i, j, 1] = 1
    else:
      categorical_train[i, j, 0] = 1

categorical_test = np.zeros((categories_test.shape[0], 7, 2))
for i in range(len(categories_test)):
  for j, string in enumerate(categories_test[i,:]):
    if string:
      categorical_test[i, j, 1] = 1
    else:
      categorical_test[i, j, 0] = 1


print(spectra_train.shape)
print(spectra_test.shape)

print(categorical_train.shape)
print(categorical_test.shape)

np.save('exp1_testspectra', spectra_test)
np.save('exp1_categorical_test', categorical_test)

# print(categorical_test.shape)
# u, c  = np.unique(categories_train, return_counts=True, axis=0)
# print(u, c)
#categories_train
#categories_train = keras.utils.to_categorical(categories_train, 128)
#categories_test = keras.utils.to_categorical(categories_test, 7)
# print(categories_train.shape)

# spectra_train = np.expand_dims(spectra_train, axis=1)
# spectra_test = np.expand_dims(spectra_test, axis=1)
# categories_test = np.expand_dims(categories_test, axis=2)
# categories_train = np.expand_dims(categories_train, axis=2)
# print(spectra_train.shape)
# spectra_train = np.tile(spectra_train, (1,7,1,1))
# spectra_test = np.tile(spectra_test, (1,7,1,1))
# print(spectra_train.shape)

# def step_func(x):
#   return (tf.math.sign(x) + 1)/2

model = dict()

for i in range(7):



  model[i] = Sequential()


  model[i].add(Conv1D(1,1, input_shape=(428,1)))
  model[i].add(BatchNormalization(center=True, scale=True))
  model[i].add(Activation('relu'))
  model[i].add(Dropout(0.5))

  # model[i].add(Flatten())
  # model[i].add(Dense(428, input_shape=(428,)))
  # model[i].add(BatchNormalization(center=True, scale=True))
  # model[i].add(Activation('relu'))
  # model[i].add(Dropout(0.5))



  # model[i].add(Conv1D(16, 8, padding='same', input_shape=(428,1)))
  # model[i].add(BatchNormalization(center=True, scale=True))
  # model[i].add(Activation('relu'))
  # model[i].add(Dropout(0.5))

  # # model[i].add(Conv1D(128,8))
  # # model[i].add(BatchNormalization(center=True, scale=True))
  # # model[i].add(Activation('relu'))
  # # model[i].add(Dropout(0.5))

  # model[i].add(MaxPooling1D(4))

  # model[i].add(Conv1D(8,4))
  # model[i].add(BatchNormalization(center=True, scale=True))
  # model[i].add(Activation('relu'))
  # model[i].add(Dropout(0.5))

  # # model[i].add(Conv1D(64,4))
  # # model[i].add(BatchNormalization(center=True, scale=True))
  # # model[i].add(Activation('relu'))
  # # model[i].add(Dropout(0.5))

  # model[i].add(MaxPooling1D(2))

  # # model[i].add(Conv1D(32,4))
  # # model[i].add(BatchNormalization(center=True, scale=True))
  # # model[i].add(Activation('relu'))
  # # model[i].add(Dropout(0.5))

  # model[i].add(Conv1D(1,1))
  # model[i].add(BatchNormalization(center=True, scale=True))
  # model[i].add(Activation('relu'))
  # model[i].add(Dropout(0.5))

  # model[i].add(MaxPooling1D(2))

  model[i].add(Flatten())
  model[i].add(Dense(200))
  model[i].add(BatchNormalization(center=True, scale=True))
  model[i].add(Activation('relu'))
  model[i].add(Dropout(0.5))

  model[i].add(Dense(100))
  model[i].add(BatchNormalization(center=True, scale=True))
  model[i].add(Activation('relu'))
  model[i].add(Dropout(0.5))

  model[i].add(Dense(50))
  model[i].add(BatchNormalization(center=True, scale=True))
  model[i].add(Activation('relu'))
  model[i].add(Dropout(0.5))

  model[i].add(Dense(10))
  model[i].add(BatchNormalization(center=True, scale=True))
  model[i].add(Activation('relu'))
  model[i].add(Dropout(0.5))

  model[i].add(Dense(2))
  model[i].add(BatchNormalization(center=True, scale=True))
  model[i].add(Activation('sigmoid'))




  model[i].compile(loss='categorical_crossentropy',
                optimizer='Nadam',
                metrics=['accuracy'])
  model[i].summary()



predict_test= np.empty((5, 7))
predict_train= np.empty((5, 7))
for num in range(1):
   print(num)
   predict_idx = np.random.randint(0,0.1*len(idx), 10)
   for i in range(7):
     hist = model[i].fit(spectra_train, categorical_train[:,i],
             batch_size=64,
             epochs=200,
             validation_data=(spectra_test, categorical_test[:,i]), shuffle=True)
     model[i].save('categorize_exp1_{}.h5'.format(i))

     # print(np.argmax(model[i].predict(spectra_test[predict_idx[:5], :,:]), axis=1))
     # print(np.argmax(categorical_test[predict_idx[:5],i], axis=1))
     predict_test[:, i] = np.argmax(model[i].predict(spectra_test[predict_idx[:5], :,:]), axis=1)
     predict_train[:, i]= np.argmax(model[i].predict(spectra_train[predict_idx[5:], :,:]), axis=1)

     print(np.sum(np.argmax(model[i].predict(spectra_test), axis=1) == np.argmax(categorical_test[:,i], axis=1))/len(categorical_test))
     save_obj(hist.history, 'history_exp1_categorize_fullrange_{}'.format(i))

   print('test')
   for j in np.arange(0,5,1):
      print(list(categories_test[predict_idx[j],:]), '-----', list(predict_test[j,:]))
   print('train')
   for j in np.arange(5,10,1):
      print(list(categories_train[predict_idx[j],:]), '-----', list(predict_train[j-5,:]))

  
