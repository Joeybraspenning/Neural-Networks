import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Flatten, MaxPooling2D, TimeDistributed
from keras.datasets import mnist
from keras.layers.normalization import BatchNormalization
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model


model = load_model('autoencoder_noise_mediannorm_005.h5')
model2 = load_model('autoencoder_noise_mediannorm_015.h5')



spectra_test = np.load('exp1_testspectra.npy')
categories = np.load('exp1_categorical_test.npy')

print(categories.shape)
print(spectra_test.shape)

categoriser = dict()
for i in range(7):
	categoriser[i] = load_model('categorize_exp1_{}.h5'.format(i))


accuracy_bare = np.zeros((100, 7))
accuracy_denoise_low = np.zeros((100, 7))
accuracy_denoise_high = np.zeros((100, 7))
#predict_test = np.empty((len(categories), 7))

for j, noise in enumerate(np.linspace(0,.25, 100)):
	spectra_test_noise = spectra_test.copy() + np.random.normal(0, noise, size=spectra_test.shape)
	spectra_denoise_low = model.predict(spectra_test_noise)
	spectra_denoise_high = model2.predict(spectra_test_noise)
	for i in range(7):

		# predict_test[:, i] = np.argmax(model.predict(spectra_test), axis=1)
		accuracy_bare[j, i] = np.sum(np.argmax(categoriser[i].predict(spectra_test_noise), axis=1) == np.argmax(categories[:,i], axis=1))/len(categories)
		accuracy_denoise_low[j, i] = np.sum(np.argmax(categoriser[i].predict(spectra_denoise_low), axis=1) == np.argmax(categories[:,i], axis=1))/len(categories)
		accuracy_denoise_high[j, i] = np.sum(np.argmax(categoriser[i].predict(spectra_denoise_high), axis=1) == np.argmax(categories[:,i], axis=1))/len(categories)

		print(i)
		print('noise = ', noise, 'accuracy noise = ', accuracy_bare[j, i])
		print('noise = ', noise, 'accuracy denoise low = ', accuracy_denoise_low[j, i])
		print('noise = ', noise, 'accuracy denoise high = ', accuracy_denoise_high[j, i])
		# print(categoriser.evaluate(spectra_test, categories[:,i]))
		# print(name[i], np.mean(hist['val_acc'][50:]))

np.save('accuracy_noise', accuracy_bare)
np.save('accuracy_denoise_005', accuracy_denoise_low)
np.save('accuracy_denoise_015', accuracy_denoise_high)

