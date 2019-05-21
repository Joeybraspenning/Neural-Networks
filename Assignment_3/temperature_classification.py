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
    with open(name + '.pkl' , 'rb') as f:
        return pickle.load(f)


# spectra = load_obj('./bachelor_data/spectra_matrix_exp3.pickle')
# categories = load_obj('./bachelor_data/input_matrix_exp3.pickle')[:,0,2]

# print(categories.shape)
# print(categories)
# print(np.unique(categories))

# print(spectra.shape)
# print(categories.shape)

# np.save('spectra_exp3', spectra)
# np.save('temperatures_exp3', categories)


spectra = np.load('spectra_exp3.npy')
temperatures = np.load('temperatures_exp3.npy')


temperatures -= np.min(temperatures)
temperatures /= np.max(temperatures)


# spectra -= np.expand_dims(np.min(spectra, axis=1), axis=1)
# print(np.sum(np.isnan(spectra)))
# print(np.sum(np.max(spectra, axis=1) == 0))
# print(np.where(np.max(spectra, axis=1) == 0))
# print(spectra[np.max(spectra, axis=1) == 0][0])
# spectra = spectra[np.max(spectra, axis=1) != 0]
# spectra = spectra.T/np.max(spectra, axis=1)


# print(categories.shape)

# print(np.max(spectra, axis=1).shape)
# median = np.median(spectra, axis=1)
# spectra = ((spectra.T - median) / np.max(spectra, axis=1)) + median
# spectra = spectra.T

# print(spectra.shape)
# print(np.unique(categories[:,1]))

idx = np.random.permutation(spectra.shape[0])
train_idx = idx[:int(0.9*len(idx))]
test_idx = idx[int(0.9*len(idx)):]
spectra_train = spectra[train_idx, :]
spectra_test = spectra[test_idx, :]
temperatures_train = np.array(temperatures[train_idx])
temperatures_test = np.array(temperatures[test_idx])
# categories_train = categories[train_idx, :]
# categories_test = categories[test_idx, :]

# categorical_train = np.zeros((categories_train.shape[0], 7, 2))
# for i in range(len(categories_train)):
#   for j, string in enumerate(categories_train[i,:]):
#     if string:
#       categorical_train[i, j, 1] = 1
#     else:
#       categorical_train[i, j, 0] = 1

# categorical_test = np.zeros((categories_test.shape[0], 7, 2))
# for i in range(len(categories_test)):
#   for j, string in enumerate(categories_test[i,:]):
#     if string:
#       categorical_test[i, j, 1] = 1
#     else:
#       categorical_test[i, j, 0] = 1


print(spectra_train.shape)
print(spectra_test.shape)

print(temperatures_train.shape)
print(temperatures_test.shape)

# np.save('exp1_testspectra', spectra_test)
# np.save('exp1_categorical_test', categorical_test)

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







model = Sequential()


drop = 0.1
# model.add(Conv1D(1,1, input_shape=(428,1)))
# model.add(BatchNormalization(center=True, scale=True))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))


model.add(Dense(428, input_shape=(428,)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))
model.add(Dropout(drop))

model.add(Dense(400))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))
model.add(Dropout(drop))

model.add(Dense(350))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))
model.add(Dropout(drop))

model.add(Dense(300))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))
model.add(Dropout(drop))

model.add(Dense(250))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))
model.add(Dropout(drop))

# model.add(Conv1D(16, 8, padding='same', input_shape=(428,1)))
# model.add(BatchNormalization(center=True, scale=True))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))

# # model.add(Conv1D(128,8))
# # model.add(BatchNormalization(center=True, scale=True))
# # model.add(Activation('relu'))
# # model.add(Dropout(0.5))

# model.add(MaxPooling1D(4))

# model.add(Conv1D(8,4))
# model.add(BatchNormalization(center=True, scale=True))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))

# # model.add(Conv1D(64,4))
# # model.add(BatchNormalization(center=True, scale=True))
# # model.add(Activation('relu'))
# # model.add(Dropout(0.5))

# model.add(MaxPooling1D(2))

# # model.add(Conv1D(32,4))
# # model.add(BatchNormalization(center=True, scale=True))
# # model.add(Activation('relu'))
# # model.add(Dropout(0.5))

# model.add(Conv1D(1,1))
# model.add(BatchNormalization(center=True, scale=True))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))

# model.add(MaxPooling1D(2))

# model.add(Flatten())
model.add(Dense(200))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))
model.add(Dropout(drop))

model.add(Dense(100))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))
model.add(Dropout(drop))

model.add(Dense(50))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))
model.add(Dropout(drop))

model.add(Dense(10))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))
model.add(Dropout(drop))

model.add(Dense(1))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))




model.compile(loss='mse',
              optimizer='Nadam',
              metrics=['accuracy'])
model.summary()



for num in range(20):
   print(num)
   predict_idx = np.random.randint(0,0.1*len(idx), 10)
   hist = model.fit(spectra_train, temperatures_train,
           batch_size=64,
           epochs=10,
           validation_data=(spectra_test, temperatures_test), shuffle=True)
   # model.save('temperature_exp3_{}.h5'.format(i))

   predict_test = model.predict(spectra_test[predict_idx[:5]])
   predict_train = model.predict(spectra_train[predict_idx[5:]])
   # print(np.argmax(model.predict(spectra_test[predict_idx[:5], :,:]), axis=1))
   # print(np.argmax(categorical_test[predict_idx[:5],i], axis=1))
   # predict_test[:, i] = np.argmax(model.predict(spectra_test[predict_idx[:5], :,:]), axis=1)
   # predict_train[:, i]= np.argmax(model.predict(spectra_train[predict_idx[5:], :,:]), axis=1)

   # print(np.sum(np.argmax(model.predict(spectra_test), axis=1) == np.argmax(categorical_test[:,i], axis=1))/len(categorical_test))
   # save_obj(hist.history, 'history_temperature_exp3')

   print('test')
   for j in np.arange(0,5,1):
      print(temperatures_test[predict_idx[j]], '-----', predict_test[j])
   print('train')
   for j in np.arange(5,10,1):
      print(temperatures_train[predict_idx[j]], '-----', predict_train[j-5])

  
