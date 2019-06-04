import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Flatten, MaxPooling2D
from keras.datasets import mnist
from keras.layers.normalization import BatchNormalization
import numpy as np
import tensorflow as tf
# import pickle

# def save_obj(obj, name ):
#     with open(name + '.pkl', 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# def load_obj(name ):
#     with open(name, 'rb') as f:
#         return pickle.load(f)


# spectra = load_obj('./bachelor_data/spectra_matrix_exp1.pickle')
# abundances = load_obj('./bachelor_data/input_matrix_exp1.pickle')[:, :, 1]


# print(np.unique(abundances[1,:]))

# print(spectra.shape)
# print(abundances.shape)

# np.save('spectra_exp1', spectra)
# np.save('abundances_exp2', abundances)

spectra = np.load('spectra.npy')
categories = np.load('categories.npy')


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
# categories_train = np.array(categories[train_idx, :], dtype='int')
# categories_test = np.array(categories[test_idx, :], dtype='int')
categories_train = categories[train_idx, :]
categories_test = categories[test_idx, :]

# categorical_train = np.zeros((categories_train.shape[0], 128))
# for i, string in enumerate(categories_train):
#    out = 0
#    for bit in list(string):
#       out = (out << 1) | bit
#    categorical_train[i, out] = 1

# categorical_test = np.zeros((categories_test.shape[0], 128))
# for i, string in enumerate(categories_test):
#    out = 0
#    for bit in list(string):
#       out = (out << 1) | bit
#    categorical_test[i, out] = 1

print(spectra_train.shape)
print(spectra_test.shape)

# print(categorical_test.shape)
# u, c  = np.unique(categories_train, return_counts=True, axis=0)
# print(u, c)
#categories_train
#categories_train = keras.utils.to_categorical(categories_train, 128)
#categories_test = keras.utils.to_categorical(categories_test, 7)
# print(categories_train.shape)

def step_func(x):
  return (tf.math.sign(x) + 1)/2

model = Sequential()

model.add(Conv1D(16, 128, input_shape=(428,1)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv1D(8,64))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(MaxPooling1D(2))

model.add(Conv1D(8,32))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv1D(4,16))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(MaxPooling1D(2))

# model.add(Conv1D(32,2))
# model.add(BatchNormalization(center=True, scale=True))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Conv1D(1,1))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# model.add(MaxPooling1D(2))

model.add(Flatten())
# model.add(Dense(100))
# model.add(BatchNormalization(center=True, scale=True))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))

# model.add(Dense(50))
# model.add(BatchNormalization(center=True, scale=True))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(25))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(7))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('sigmoid'))




model.compile(loss='mse',
              optimizer='Nadam',
              metrics=['accuracy'])
model.summary()

for i in range(1000):
   print(i)
   hist = model.fit(spectra_train, categories_train,
           batch_size=64,
           epochs=1,
           validation_data=(spectra_test, categories_test), shuffle=True)
   predict_idx = np.random.randint(0,0.1*len(idx), 10)
   predict_test = model.predict(spectra_test[predict_idx[:5], :,:])
   predict_train = model.predict(spectra_train[predict_idx[5:], :,:])

   print('test')
   for j in np.arange(0,5,1):
      print(list(categories_test[predict_idx[j],:]), '-----', list(predict_test[j]))
   print('train')
   for j in np.arange(5,10,1):
      print(list(categories_train[predict_idx[j],:]), '-----', list(predict_train[j-5]))
