import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.datasets import mnist
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)


spectra = load_obj('./bachelor_data/spectra_matrix_exp2.pickle')
categories = load_obj('./bachelor_data/input_matrix_exp2.pickle')[:, :, 0]

print(spectra.shape)
print(categories.shape)

idx = np.random.permutation(spectra.shape[0])
train_idx = idx[:int(0.9*len(idx))]
test_idx = idx[int(0.9*len(idx)):]
spectra_train = spectra[train_idx, :]
spectra_test = spectra[test_idx, :]
categories_train = categories[train_idx, :]
categories_test = categories[test_idx, :]

#categories_train = keras.utils.to_categorical(categories_train)
#categories_test = keras.utils.to_categorical(categories_test, 7)
print(spectra_train.shape)

model = Sequential()

model.add(Dense(428, input_shape = (428,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(BatchNormalization())
model.add(Dense(300))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(BatchNormalization())
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(BatchNormalization())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(BatchNormalization())
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(BatchNormalization())
model.add(Dense(7))
model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',
              optimizer='Nadam',
              metrics=['accuracy'])
model.summary()

for i in range(100):
   hist = model.fit(spectra_train, categories_train,
           batch_size=16,
           epochs=1,
           validation_data=(spectra_test, categories_test))
   predict_idx = np.random.randint(0,0.1*len(idx), 10)
   predict = model.predict(spectra_test[predict_idx, :])
   for j in range(10):
      print(list(categories_test[predict_idx[j], :]), '-----', list(predict[j]))
