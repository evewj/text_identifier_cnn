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

refresh_file_structure()

def rotate_image(image, angle, scale):
  path =  os.path.dirname(os.path.abspath(__file__))+ "/content/"
  background = cv2.imread(path+'blank_plate.png')[85:235,50:150]
  shape= (image.shape[1], image.shape[0])
  image_center = tuple(np.array(shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(220,220,220))
  #result = cv2.resize(result, shape)
  return result #+ background

def add_random_plates(num_plates, blur_max=1):
  for i in range(0, num_plates):

    # Pick two random letters
    plate_alpha = ""
    for _ in range(0, 2):
        plate_alpha += (random.choice(string.ascii_uppercase))

    # Pick two random numbers
    num = randint(0, 99)
    plate_num = "{:02d}".format(num)

    blur_amount = randint(1,blur_max)

    add_plate(plate_alpha, plate_num, blur_amount)


def add_custom_plates(plate_amount, plate_alpha, plate_num, blur_max=1, noise_max = 0):
    blur_amount = randint(1,blur_max)

    add_plate(plate_alpha, plate_num, blur_amount)


def add_plate(plate_alpha, plate_num, blur_amount=1, noise_amount = 0, darkness_amount = 0.25,  rotation_threshold = 20, scale_threshold = 0.25):
    path =  os.path.dirname(os.path.abspath(__file__))+ "/content/"

    # Write plate to image
    blank_plate = cv2.imread(path+'blank_plate.png')

    # Convert into a PIL image (this is so we can use the monospaced fonts)
    blank_plate_pil = Image.fromarray(blank_plate)

    # Get a drawing context
    draw = ImageDraw.Draw(blank_plate_pil)
    monospace = ImageFont.truetype(font="/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                                   size=165)
    draw.text(xy=(48, 75),
              text=plate_alpha + " " + plate_num,
              fill=(0,0,0), font=monospace)

    # Convert back to OpenCV image and save
    blank_plate = np.array(blank_plate_pil)

    char1 = blank_plate[85:235,50:150]
    char2 = blank_plate[85:235,150:250]
    char3 = blank_plate[85:235,350:450]
    char4 = blank_plate[85:235,450:550]

    angle = random.uniform(-1*rotation_threshold, rotation_threshold)
    scale = random.uniform(scale_threshold, 1)

    blur_amount = int(blur_amount*scale) +1
    char1 = rotate_image(char1, angle, scale)*random.uniform(darkness_amount, 1)
    char2 = rotate_image(char2, angle, scale)*random.uniform(darkness_amount, 1)
    char3 = rotate_image(char3, angle, scale)*random.uniform(darkness_amount, 1)
    char4 = rotate_image(char4, angle, scale)*random.uniform(darkness_amount, 1)

    char1 = cv2.blur(char1,(blur_amount,blur_amount))
    char2 = cv2.blur(char2,(blur_amount,blur_amount))
    char3 = cv2.blur(char3,(blur_amount,blur_amount))
    char4 = cv2.blur(char4,(blur_amount,blur_amount))

    # Write license plate to file
    cv2.imwrite(os.path.join(path + "pictures/" + plate_alpha[0] + "/",
                             "plate_{}{}.png".format(plate_alpha, plate_num)),
                             char1)
    cv2.imwrite(os.path.join(path + "pictures/" + plate_alpha[1] + "/",
                             "plate_{}{}.png".format(plate_alpha, plate_num)),
                             char2)
    cv2.imwrite(os.path.join(path + "pictures/" + plate_num[0] + "/",
                             "plate_{}{}.png".format(plate_alpha, plate_num)),
                             char3)
    cv2.imwrite(os.path.join(path + "pictures/" + plate_num[1] + "/",
                             "plate_{}{}.png".format(plate_alpha, plate_num)),
                             char4)

add_random_plates(10000, blur_max=35)


PATH = os.path.dirname(os.path.abspath(__file__))+ "/content/pictures/"

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

char_lookup = {}
count = 0

for char in alphabet:
  char_lookup[char] = count
  count+=1

char_inverse_lookup = dict(map(reversed, char_lookup.items()))

#import pictures
folderA = PATH + 'A/*'
filesA = glob.glob(folderA)
all_dataset = np.array([[np.array(cv2.imread(file, cv2.IMREAD_GRAYSCALE)), char_lookup['A']] for file in filesA[:]])

for char in alphabet:
  folder = PATH + char + '/*'
  files = glob.glob(folder)
  new_dataset = np.array([[np.array(cv2.imread(file, cv2.IMREAD_GRAYSCALE)), char_lookup[char]] for file in files[:]])
  all_dataset = np.concatenate((all_dataset, new_dataset), axis=0)


np.random.shuffle(all_dataset)
X_dataset_orig = np.array([data[0] for data in all_dataset[:]])
Y_dataset_orig = np.array([[data[1]] for data in all_dataset]).T

NUMBER_OF_LABELS = 36
CONFIDENCE_THRESHOLD = 0.01

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

# Normalize X (images) dataset
X_dataset = X_dataset_orig/255.

shape = X_dataset.shape
X_dataset = X_dataset.reshape(shape[0], shape[1], shape[2], 1)

# Convert Y dataset to one-hot encoding
Y_dataset = convert_to_one_hot(Y_dataset_orig, NUMBER_OF_LABELS).T

VALIDATION_SPLIT = 0.2

print("Total examples: {:d}\nTraining examples: {:f}\nTest examples: {:f}".format(X_dataset.shape[0], math.ceil(X_dataset.shape[0] * (1-VALIDATION_SPLIT)), math.floor(X_dataset.shape[0] * VALIDATION_SPLIT)))
print("X shape: " + str(X_dataset.shape))
print("Y shape: " + str(Y_dataset.shape))

conv_model = models.Sequential()
conv_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 100,1)))
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

conv_model.save(os.path.dirname(os.path.abspath(__file__)) + '/my_model')

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

saved_model = models.load_model(os.path.dirname(os.path.abspath(__file__)) + '/my_model')
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


for i in range (50,55):
  displayImage(i)


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
  y_predict = conv_model.predict(img_aug)[0]

  prediction = char_inverse_lookup[np.argmax(y_predict)]
  truth = char_inverse_lookup[np.argmax(Y_dataset[index])]

  test_labels.append(truth)
  predictions.append(prediction)


cm = confusion_matrix(y_true=test_labels, y_pred=predictions)

cm_plot_labels = [char for char in alphabet]

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix', normalize=True)
