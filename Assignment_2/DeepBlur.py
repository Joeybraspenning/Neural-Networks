from __future__ import print_function

import os
from keras.preprocessing import image as image_utils
from PIL import Image, ImageFilter
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Reshape, Conv2D
from keras.layers import MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import load_model

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

os.chdir('./data')
image_matrix = np.load('caltech_datacube.npy')
blur_matrix = np.load('caltech_blurcube.npy')

image_matrix = image_matrix.reshape(image_matrix.shape[2], 100, 100, 1)
blur_matrix = blur_matrix.reshape(blur_matrix.shape[2], 100, 100, 1)
os.chdir('..')


np.random.seed(123)
train_idx = np.random.permutation(30607)[:250]
test_idx = np.random.permutation(30607)[250:300]


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', padding='same',
                 input_shape=(100,100, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(2000, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(10000, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Reshape((100,100,1)))

model.compile(optimizer='adam', \
				loss = 'binary_crossentropy', metrics=['accuracy'])
				
history = model.fit(blur_matrix[train_idx,:,:,:], image_matrix[train_idx,:,:,:],\
					batch_size = 4,\
					epochs = 30,\
					validation_data = (blur_matrix[test_idx,:,:,:], image_matrix[test_idx,:,:,:]),
					verbose = True,
					shuffle = True)





