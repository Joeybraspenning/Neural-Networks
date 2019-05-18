import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, UpSampling1D, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
# import matplotlib.pyplot as plt
import numpy as np


spectra = np.load('spectra_exp1.npy')
spectra = spectra.T/np.max(spectra, axis=1)
#median = np.mean(spectra, axis=1)
#spectra = ((spectra.T - median) / np.max(spectra, axis=1)) + median
spectra = spectra.T

idx = np.random.permutation(spectra.shape[0])
train_idx = idx[:int(0.9*len(idx))]
test_idx = idx[int(0.9*len(idx)):]
spectra_train = np.expand_dims(spectra[train_idx, :], axis=2)
spectra_test = np.expand_dims(spectra[test_idx, :], axis=2)

spectra_train_noise = spectra_train #+ np.random.normal(0, 0.001, size=spectra_train.shape)
spectra_test_noise = spectra_test #+ np.random.normal(0, 0.001, size=spectra_test.shape)


#######################################################


# print(spectra_train_noise[:,0])
print(spectra_test.shape)
name = 'relu'
model = Sequential()
# model.add(Dense(200, input_shape = (428,)))
# model.add(BatchNormalization(center=True, scale=True))
# model.add(Activation(name))
# model.add(Dropout(0.2))


model.add(Conv1D(64, 3, strides=2, padding='same', input_shape=(428, 1)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))
# model.add(Dropout(0.2))

model.add(Conv1D(32, 3, strides=2, padding='same'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))

model.add(Conv1D(16, 3, strides=2, padding='same'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))

model.add(Conv1D(8, 7, strides=3, padding='valid'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(16))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))

model.add(Reshape((16, 1)))

# model.add(UpSampling1D(size=2))
model.add(Conv1D(8, 3, padding='same'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))

model.add(UpSampling1D(size=7))
model.add(Conv1D(16, 5, strides=2, padding='valid'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))

model.add(UpSampling1D(size=2))
model.add(Conv1D(32, 3, padding='same'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))

model.add(UpSampling1D(size=2))
model.add(Conv1D(64, 3, padding='same'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))

model.add(UpSampling1D(size=2))
model.add(Conv1D(1, 5, padding='valid'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('relu'))


# model.add(Dense(100))
# model.add(BatchNormalization(center=True, scale=True))
# model.add(Activation(name))
# model.add(Dropout(0.2))

# model.add(Dense(50))
# model.add(BatchNormalization(center=True, scale=True))
# model.add(Activation(name))
# model.add(Dropout(0.2))

# model.add(Dense(100))
# model.add(BatchNormalization(center=True, scale=True))
# model.add(Activation(name))
# model.add(Dropout(0.2))

# model.add(Dense(200))
# model.add(BatchNormalization(center=True, scale=True))
# model.add(Activation(name))
# model.add(Dropout(0.2))


# model.add(Dense(428))
# model.add(BatchNormalization(center=True, scale=True))
# model.add(Activation('sigmoid'))

model.compile(optimizer='Nadam', \
				loss = 'mse', metrics=['accuracy'])

model.summary()

history = model.fit(spectra_train_noise, spectra_train,\
					batch_size = 128,\
					epochs = 300,\
					validation_data = (spectra_test_noise, spectra_test),
					verbose = True,
					shuffle = True)
model.save('autoencoder.h5')



# model = load_model('autoencoder.h5')



# prediction = model.predict(spectra_test_noise)


# plt.plot(prediction[0,:], label='prediction')
# plt.plot(spectra_test[0,:], label='true')
# plt.legend()
# plt.show()
'''
for tel, i in enumerate(np.where(Y_test[:, 6] == 1 )[0][0:10]):
	fig, ax = plt.subplots(1, 4)
	ax[0].imshow(X_test[i].reshape(28, 28), cmap = 'Greys')

	ax[1].imshow(X_test_noise[i].reshape(28, 28), cmap = 'Greys')

	ax[2].imshow(prediction[i].reshape(28, 28), cmap = 'Greys')

	ax[3].imshow(encodes[i].reshape(10, 5), cmap = 'Greys')

	ax[0].set_xticks([])
	ax[1].set_xticks([])
	ax[2].set_xticks([])
	ax[3].set_xticks([])


	ax[0].set_yticks([])
	ax[1].set_yticks([])
	ax[2].set_yticks([])
	ax[3].set_yticks([])


	plt.tight_layout()
	plt.savefig('auto_6_{}.png'.format(tel))
	plt.show()
'''
# with open('{}_keras_100_50_BN_DO_hist'.format(name), 'wb') as file_pi:
#     pickle.dump(history.history, file_pi, pickle.HIGHEST_PROTOCOL)
