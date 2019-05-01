import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.datasets import mnist
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import numpy as np

'''
abundance = np.empty((1600,2))
spectra = np.empty((1600,1204))
for i in range(1600):
   abundance[i,:] = np.loadtxt('./output_k_na/abunds_{}.txt'.format(i))
   spectra[i,:] = np.loadtxt('./input_k_na/spectrum_{}.txt'.format(i))

np.save('spectra', spectra)
np.save('abundances', abundance)
'''

spectra = np.load('spectra.npy')
abundance = -np.log10(np.load('abundances.npy'))

idx = np.random.permutation(1600)
train_idx = idx[:1500]
test_idx = idx[1500:]
spectra_train = np.expand_dims(spectra[train_idx, :], axis=2)
spectra_test = np.expand_dims(spectra[test_idx, :], axis=2)
abundance_train = abundance[train_idx, :]
abundance_test = abundance[test_idx, :]

print(spectra_train.shape)

model = Sequential()

model.add(Conv1D(64, 128, input_shape=(1204,1)))
model.add(Activation('relu'))


model.add(BatchNormalization())
model.add(Conv1D(32,64))
model.add(Activation('relu'))


model.add(BatchNormalization())
model.add(MaxPooling1D(4))

model.add(Conv1D(1, 1, padding='same'))
model.add(Flatten())
model.add(Dense(60))
model.add(Activation('relu'))

model.add(BatchNormalization())
model.add(Dense(10))
model.add(Activation('relu'))

model.add(BatchNormalization())
model.add(Dense(2))
model.add(Activation('relu'))



model.compile(loss='mse',
              optimizer='Nadam',
              metrics=['accuracy'])
model.summary()

for i in range(100):
   hist = model.fit(spectra_train, abundance_train,
           batch_size=16,
           epochs=1,
           validation_data=(spectra_test, abundance_test))
   predict_idx = np.random.randint(0,100, 10)
   predict = model.predict(spectra_test[predict_idx, :,:])
   for j in range(10):
      print(abundance_test[predict_idx[j],:], '-----', predict[j])
