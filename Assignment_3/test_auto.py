import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Flatten, MaxPooling2D, TimeDistributed
from keras.datasets import mnist
from keras.layers.normalization import BatchNormalization
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model


model = load_model('autoencoder_noise_mediannorm_lessnoisefirst.h5')
modell2 = load_model('autoencoder_noise_mediannorm_noisefirst.h5')



spectra_test = np.load('exp1_testspectra.npy')
categories = np.load('exp1_categorical_test.npy')

print(categories.shape)
print(spectra_test.shape)

predict_test= np.empty((len(categories), 7))
for noise in np.linspace(0,0.5, 5):
	spectra_test_noise = spectra_test.copy() + np.random.normal(0, noise, size=spectra_test.shape)
	for i in range(7):
		categoriser = load_model('categorize_exp1_{}.h5'.format(i))

		# predict_test[:, i] = np.argmax(model.predict(spectra_test), axis=1)

		print('noise = ', noise, 'accuracy = ', np.sum(np.argmax(categoriser.predict(spectra_test_noise), axis=1) == np.argmax(categories[:,i], axis=1))/len(categories))
		# print(categoriser.evaluate(spectra_test, categories[:,i]))
		# print(name[i], np.mean(hist['val_acc'][50:]))


