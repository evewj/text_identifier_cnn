#!/usr/bin/env python
import tensorflow as tf
import cv2
import os
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob
import numpy as np

def noisy_pixelate(img):
  do_rescale = True if np.random.rand() > 0.5 else False
  do_noise = True if np.random.rand() > 0.5 else False

  out = img
  if do_rescale:
    width = np.random.randint(20,100)
    height = np.random.randint(20,100)

    small = cv2.resize(img, (width,height), interpolation=cv2.INTER_LINEAR)
    small = small + np.random.randint(np.random.randint(-20, high=0),np.random.randint(1,20),size=small.shape)
    out = cv2.resize(small, (100,150),interpolation=cv2.INTER_NEAREST)
  if do_noise:
    out = out + np.random.randint(np.random.randint(-20, high=0),np.random.randint(1,20),size=out.shape)

  return np.clip(out,0,255)

def refresh_file_structure():
   # Run this command to refresh file structure
   alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
   folder =  os.path.dirname(os.path.abspath(__file__))+ '/data/augmented/'
   for filename in os.listdir(folder):
       file_path = os.path.join(folder, filename)
       try:
           if os.path.isfile(file_path) or os.path.islink(file_path):
               os.unlink(file_path)
           elif os.path.isdir(file_path):
               shutil.rmtree(file_path)
       except Exception as e:
           print('Failed to delete %s. Reason: %s' % (file_path, e))

   for char in alphabet:
     dir = folder + char
     os.mkdir(dir)

refresh_file_structure()

batch_size = 16

# this is the augmentation configuration we will use for training
datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        brightness_range= (0.05,1.5),
        horizontal_flip=False,
        fill_mode='nearest',
        preprocessing_function=noisy_pixelate)


PATH = os.path.dirname(os.path.abspath(__file__))+ "/data/input/"

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'#'0123456789' # '0123456789'

char_lookup = {}
count = 0

for char in alphabet:
  char_lookup[char] = count
  count+=1

char_inverse_lookup = dict(map(reversed, char_lookup.items()))

for char in alphabet:
    print(char)
    this_path = os.path.dirname(os.path.abspath(__file__)) + '/data/augmented/' + char
    folder = PATH + char + '/*'
    files = glob.glob(folder)
    for file in files:
      img = cv2.imread(file,cv2.IMREAD_COLOR)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = np.expand_dims(img, axis=0)
      i = 0
      for batch in datagen.flow(img, batch_size=1,
                          save_to_dir=this_path, save_prefix=char, save_format='png'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely
