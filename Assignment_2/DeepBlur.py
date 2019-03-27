import os
from keras.preprocessing import image as image_utils
from PIL import Image, ImageFilter
import numpy as np

image_matrix = np.zeros((200, 200, 30607))
counter = 0
os.chdir('./data/256_ObjectCategories')
for dirs in os.listdir(os.getcwd()):
   os.chdir(dirs)
   print(dirs)
   for files in os.listdir(os.getcwd()):

      I = image_utils.load_img(files, target_size=(200,200))
      J = image_utils.img_to_array(I)
      J /= np.max(J)
      img = Image.fromarray(np.uint8(J*255))
      image_matrix[:, :, counter] = np.sum(img, axis=2)
      counter +=1
      print(counter)
   os.chdir('..')

np.save('caltech_datacube', image_matrix)
      

