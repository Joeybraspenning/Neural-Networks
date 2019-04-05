from __future__ import print_function

import os
from keras.preprocessing import image as image_utils
#from PIL import Image, ImageFilter
import numpy as np
import keras
from keras.models import Sequential
import keras.layers
from keras.layers import Dense, Activation, Dropout, Reshape, Conv2D
from keras.layers import MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import load_model


#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="2"


'''
image_matrix = np.zeros((100, 100, 30607))
blur_matrix = np.zeros((100, 100, 30607))
counter = 0
os.chdir('./data/256_ObjectCategories')
for dirs in os.listdir(os.getcwd()):
   os.chdir(dirs)
   print(dirs)
   for files in os.listdir(os.getcwd()):

      I = image_utils.load_img(files, target_size=(100,100))
      #J = image_utils.img_to_array(I)
      #J /= np.max(J)
     # img = Image.fromarray(np.uint8(J*255))
      image_matrix[:, :, counter] = np.sum(I, axis=2)/(255.*3)
      blur_matrix[:, :, counter] = np.sum(I.filter(ImageFilter.GaussianBlur), axis=2)/(255.*3)
      counter +=1
      if np.mod(counter, 10) == 0:
         print(counter, end='\r')
   os.chdir('..')

np.save('caltech_datacube', image_matrix)
np.save('caltech_blurcube', blur_matrix)
'''

#os.chdir('./data')
image_matrix = np.load('caltech_datacube.npy')
blur_matrix = np.load('caltech_blurcube.npy')

image_matrix = np.swapaxes(np.expand_dims(image_matrix, axis=3), 0, 2)
blur_matrix = np.swapaxes(np.expand_dims(blur_matrix, axis=3), 0, 2)

# image_matrix = np.swapaxes(image_matrix, 0, 2)
# blur_matrix = np.swapaxes(blur_matrix, 0, 2)


# image_matrix = image_matrix.reshape(image_matrix.shape[2], 100, 100, 1)
# blur_matrix = blur_matrix.reshape(blur_matrix.shape[2], 100, 100, 1)
#os.chdir('..')


np.random.seed(123)
train_idx = np.random.permutation(30607)[:500]
test_idx = np.random.permutation(30607)[500:550]

'''
#model = Sequential()
#model.add(Conv2D(32, kernel_size=(3, 3),
#                 padding='same',
#                 input_shape=(5,5, 1)))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Dropout(0.25))

#model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(Dropout(0.25))

#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(400))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(25))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))


model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(400))
model.add(BatchNormalization())
model.add(Activation('softmax'))

model.add(Reshape((20,20,1)))
'''
'''
RNN = keras.layers.LSTM
LAYERS = 1

model=Sequential()

model.add(RNN(256, input_shape=(100,100)))

model.add(keras.layers.RepeatVector(100))

for _ in range(LAYERS):
	model.add(RNN(256, return_sequences=True))

model.add(keras.layers.TimeDistributed(Dense(100, activation = 'softmax')))

model.compile(optimizer='adam', \
            loss = 'mean_squared_error', metrics=['accuracy'])

model.summary()

history = model.fit(blur_matrix[train_idx,:,:], image_matrix[train_idx,:,:],\
               batch_size = 32,\
               epochs = 100,\
               validation_data = (blur_matrix[test_idx,:,:], image_matrix[test_idx,:,:]),
               verbose = True,
               shuffle = True)
model.save('Deepblur.h5')

'''
model = Sequential()

model.add(Conv2D(64, 9, input_shape=(100, 100, 1), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))


model.add(Conv2D(32, 1, padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Conv2D(1, 5, padding='same'))
model.add(Activation('sigmoid'))



model.compile(optimizer='adam', \
				loss = 'mean_squared_error', metrics=['accuracy'])

model.summary()
				
history = model.fit(blur_matrix[train_idx,:,:,:], image_matrix[train_idx,:,:,:],\
					batch_size = 16,\
					epochs = 50,\
					validation_data = (blur_matrix[test_idx,:,:,:], image_matrix[test_idx,:,:,:]),
					verbose = True,
					shuffle = True)

model.save('Deepblur.h5')


'''
model = load_model('Deepblur.h5')

import matplotlib.pyplot as plt
import tensorflow as tf


def mean_squared_error(true, pred):
  """L2 distance between tensors true and pred.
  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
  return np.sum((true - pred)**2) / float(pred.size)


def psnr(true, pred):
  """Image quality metric based on maximal signal power vs. power of the noise.
  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    peak signal to noise ratio (PSNR)
  """
  return 10.0 * np.log(1.0 / mean_squared_error(true, pred)) / np.log(10.0)


prediction = model.predict(blur_matrix[:16, :, :, :])
for x in range(16):
   print(psnr(image_matrix[x,:,:,0], prediction[x,:,:,0]))
   fig,ax = plt.subplots(1,3)
   ax[0].imshow(blur_matrix[x,:,:,0].reshape((100, 100)), cmap='Greys')
   ax[1].imshow(image_matrix[x,:,:,0].reshape((100, 100)), cmap='Greys')
   ax[2].imshow(prediction[x,:,:,0].reshape((100, 100)), cmap='Greys')
   plt.show()
'''