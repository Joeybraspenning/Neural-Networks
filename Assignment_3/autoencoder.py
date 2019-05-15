import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np


spectra = np.load('spectra_exp1.npy')
spectra = spectra.T/np.max(spectra, axis=1)
#median = np.mean(spectra, axis=1)
#spectra = ((spectra.T - median) / np.max(spectra, axis=1)) + median
spectra = spectra.T

idx = np.random.permutation(spectra.shape[0])
train_idx = idx[:int(0.9*len(idx))]
test_idx = idx[int(0.9*len(idx)):]
spectra_train = spectra[train_idx, :].astype('float32')
spectra_test = spectra[test_idx, :].astype('float32')

spectra_train_noise = spectra_train #+ np.random.normal(0, 0.001, size=spectra_train.shape)
spectra_test_noise = spectra_test #+ np.random.normal(0, 0.001, size=spectra_test.shape)


#######################################################

'''
print(spectra_train_noise[:,0])

name = 'relu'
model = Sequential()
model.add(Dense(200, input_shape = (428,)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation(name))
model.add(Dropout(0.2))

# model.add(Dense(900, activation=name))
# model.add(Dense(800, activation=name))
# model.add(Dense(700, activation=name))
# model.add(Dense(600, activation=name))
# model.add(Dense(500, activation=name))
# model.add(Dense(400, activation=name))
# model.add(Dense(300, activation=name))
# model.add(Dense(200, activation=name))
# model.add(Dense(100, activation=name))
model.add(Dense(100))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation(name))
model.add(Dropout(0.2))

model.add(Dense(50))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation(name))
model.add(Dropout(0.2))

model.add(Dense(100))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation(name))
model.add(Dropout(0.2))

model.add(Dense(200))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation(name))
model.add(Dropout(0.2))


model.add(Dense(428))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('sigmoid'))

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
'''


model = load_model('autoencoder.h5')



prediction = model.predict(spectra_test_noise)


plt.plot(prediction[0,:], label='prediction')
plt.plot(spectra_test[0,:], label='true')
plt.legend()
plt.show()
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
