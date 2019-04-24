import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import matplotlib.pyplot as plt
import pickle
import numpy as np

N_categories = 10
N_features = 28*28

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], N_features).astype('float32')
X_test = X_test.reshape(X_test.shape[0], N_features).astype('float32')
print(X_test.shape, X_test[0].shape)
X_train /= 255
X_test /= 255

Y_train = keras.utils.to_categorical(Y_train, N_categories)
Y_test = keras.utils.to_categorical(Y_test, N_categories)

X_train_noise = X_train + np.random.normal(0.5, 0.8, size = X_train.shape)
X_test_noise = X_test + np.random.normal(0.5, 0.8, size = X_test.shape)

#######################################################




name = 'relu'
model = Sequential()
model.add(Dense(200, input_shape = (N_features,)))
model.add(BatchNormalization())
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
model.add(BatchNormalization())
model.add(Activation(name))
model.add(Dropout(0.2))

model.add(Dense(50))
model.add(BatchNormalization())
model.add(Activation(name))
model.add(Dropout(0.2))

model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation(name))
model.add(Dropout(0.2))

model.add(Dense(200))
model.add(BatchNormalization())
model.add(Activation(name))
model.add(Dropout(0.2))

# model.add(Dense(25, activation=name))
# model.add(Dense(50, activation='sigmoid'))
model.add(Dense(N_features))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', \
				loss = 'binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_noise, X_train,\
					batch_size = 256,\
					epochs = 30,\
					validation_data = (X_test_noise, X_test),
					verbose = True,
					shuffle = True)
model.save('autoencoder.h5')
'''


model = load_model('autoencoder.h5')

model2 = Sequential()
for j in range(11):
	model2.add(model.layers[j])

# print(model2.predict(X_test_noise)[0])
print(model.layers[10])

prediction = model.predict(X_test_noise)
encodes = model2.predict(X_test_noise)
print(Y_test[:, 6])
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

# with open('{}_keras_100_50_BN_DO_hist'.format(name), 'wb') as file_pi:
#     pickle.dump(history.history, file_pi, pickle.HIGHEST_PROTOCOL)
'''