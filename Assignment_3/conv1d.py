import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.datasets import mnist
from keras.layers.normalization import BatchNormalization
import numpy as np
import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)


# spectra = load_obj('./bachelor_data/spectra_matrix_exp2.pickle')
# categories = load_obj('./bachelor_data/input_matrix_exp2.pickle')[:, :, 0]

# print(spectra.shape)
# print(categories.shape)

# np.save('spectra', spectra)
# np.save('categories', categories)

spectra = np.load('spectra.npy')
categories = np.load('categories.npy')

idx = np.random.permutation(spectra.shape[0])
train_idx = idx[:int(0.9*len(idx))]
test_idx = idx[int(0.9*len(idx)):]
spectra_train = np.expand_dims(spectra[train_idx, :], axis=2)
spectra_test = np.expand_dims(spectra[test_idx, :], axis=2)
categories_train = np.array(categories[train_idx, :], dtype='int')
categories_test = np.array(categories[test_idx, :], dtype='int')

categorical_train = np.zeros((categories_train.shape[0], 128))
for i, string in enumerate(categories_train):
   out = 0
   for bit in list(string):
      out = (out << 1) | bit
   categorical_train[i, out] = 1

categorical_test = np.zeros((categories_test.shape[0], 128))
for i, string in enumerate(categories_test):
   out = 0
   for bit in list(string):
      out = (out << 1) | bit
   categorical_test[i, out] = 1

print(spectra_train.shape)
print(spectra_test.shape)

#categories_train
#categories_train = keras.utils.to_categorical(categories_train, 128)
#categories_test = keras.utils.to_categorical(categories_test, 7)
print(categories_train.shape)

model = Sequential()

model.add(Conv1D(64, 32, padding='same', input_shape=(428,1)))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(BatchNormalization())
model.add(Conv1D(32,16, padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(BatchNormalization())
model.add(MaxPooling1D(2))

model.add(Conv1D(1, 1, padding='same'))
model.add(Flatten())
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(BatchNormalization())
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(BatchNormalization())
model.add(Dense(128))
model.add(Activation('sigmoid'))




model.compile(loss='categorical_crossentropy',
              optimizer='Nadam',
              metrics=['accuracy'])
model.summary()

for i in range(100):
   hist = model.fit(spectra_train, categorical_train,
           batch_size=16,
           epochs=1,
           validation_data=(spectra_test, categorical_test))
   predict_idx = np.random.randint(0,0.1*len(idx), 10)
   predict = model.predict(spectra_test[predict_idx, :,:])
   for j in range(10):
      print(categories_test[predict_idx[j],:], '-----', predict[j])
