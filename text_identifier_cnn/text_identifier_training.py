#!/usr/bin/env python
import string
import random
from random import randint
import cv2
import numpy as np
import os
import shutil
import re
import math
import glob
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from keras import layers
from keras import models
from keras import optimizers

from keras.utils import plot_model
from keras import backend

from sklearn.metrics import confusion_matrix
import itertools

def refresh_file_structure():
  # Run this command to refresh file structure
  alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
  folder =  os.path.dirname(os.path.abspath(__file__))+ '/content/pictures/'
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

#refresh_file_structure()

PATH = os.path.dirname(os.path.abspath(__file__))+ "/content/pictures/"

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'#'0123456789' # '0123456789'

char_lookup = {}
count = 0

for char in alphabet:
  char_lookup[char] = count
  count+=1

char_inverse_lookup = dict(map(reversed, char_lookup.items()))

#import pictures
folderA = PATH + 'A/*'
#folderA = PATH + '0/*'
filesA = glob.glob(folderA)
#all_dataset = np.array([[np.array(cv2.imread(file, cv2.IMREAD_GRAYSCALE)), char_lookup['0']] for file in filesA[:]])
all_dataset = np.array([[np.array(cv2.imread(file, cv2.IMREAD_COLOR)), char_lookup['A']] for file in filesA[:]])
for char in alphabet:
  folder = PATH + char + '/*'
  files = glob.glob(folder)
  print(char)
  new_dataset = np.array([[np.array(cv2.imread(file, cv2.IMREAD_COLOR)), char_lookup[char]] for file in files[:]])
  all_dataset = np.concatenate((all_dataset, new_dataset), axis=0)


np.random.shuffle(all_dataset)
X_dataset_orig = np.array([data[0] for data in all_dataset[:]])
Y_dataset_orig = np.array([[data[1]] for data in all_dataset]).T

#NUMBER_OF_LABELS = 26
NUMBER_OF_LABELS = 36
#NUMBER_OF_LABELS = 36
CONFIDENCE_THRESHOLD = 0.01

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

# Normalize X (images) dataset
X_dataset = X_dataset_orig/255.

#shape = X_dataset.shape
#X_dataset = X_dataset.reshape(shape[0], shape[1], shape[2], 1)

# Convert Y dataset to one-hot encoding
Y_dataset = convert_to_one_hot(Y_dataset_orig, NUMBER_OF_LABELS).T

VALIDATION_SPLIT = 0.2

print("Total examples: {:d}\nTraining examples: {:f}\nTest examples: {:f}".format(X_dataset.shape[0], math.ceil(X_dataset.shape[0] * (1-VALIDATION_SPLIT)), math.floor(X_dataset.shape[0] * VALIDATION_SPLIT)))
print("X shape: " + str(X_dataset.shape))
print("Y shape: " + str(Y_dataset.shape))

conv_model = models.Sequential()
conv_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 100,3)))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Flatten())
conv_model.add(layers.Dropout(0.5))
conv_model.add(layers.Dense(512, activation='relu'))
conv_model.add(layers.Dense(36, activation='softmax'))

conv_model.summary()

LEARNING_RATE = 1e-4
conv_model.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                   metrics=['acc'])

history_conv = conv_model.fit(X_dataset, Y_dataset,
                              validation_split=VALIDATION_SPLIT,
                              epochs=20,
                              batch_size=16)

conv_model.save(os.path.dirname(os.path.abspath(__file__)) + '/model2')
#conv_model.save(os.path.dirname(os.path.abspath(__file__)) + '/letter_model')

plt.plot(history_conv.history['loss'])
plt.plot(history_conv.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'], loc='upper left')
plt.show()

plt.plot(history_conv.history['acc'])
plt.plot(history_conv.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy (%)')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
plt.show()

saved_model = models.load_model(os.path.dirname(os.path.abspath(__file__)) + '/model2')
#saved_model = models.load_model(os.path.dirname(os.path.abspath(__file__)) + '/letter_model')
# Display images in the training data set.
def displayImage(index):
  img = X_dataset[index]

  img_aug = np.expand_dims(img, axis=0)
  y_predict = saved_model.predict(img_aug)[0]

  #plt.imshow(img)
  prediction = char_inverse_lookup[np.argmax(y_predict)]
  truth = char_inverse_lookup[np.argmax(Y_dataset[index])]
  caption = ("Truth: {} | Predicted: {}".
             format(truth, prediction,))
  print(caption)


test_labels = []
predictions = []

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '',
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.figure(figsize=(10,10))
    plt.show()


for index in range(len(X_dataset)):
  img = X_dataset[index]
  img_aug = np.expand_dims(img, axis=0)
  y_predict = saved_model.predict(img_aug)[0]

  prediction = char_inverse_lookup[np.argmax(y_predict)]
  truth = char_inverse_lookup[np.argmax(Y_dataset[index])]

  test_labels.append(truth)
  predictions.append(prediction)


cm = confusion_matrix(y_true=test_labels, y_pred=predictions)

cm_plot_labels = [char for char in alphabet]

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix', normalize=True)
