###  Mini project 3
### McGill University
### Group 16

'''
Step 1: Load Data
'''
# ref: https://www.kaggle.com/comp551f2018ta/example-to-load-data
import numpy as np
import pandas as pd
import os

cwd = os.getcwd()
input_path = cwd+"/data"
print(os.listdir(input_path))

# Any results you write to the current directory are saved as output.

#Files are stored in pickle format.
#Load them like how you load any pickle. The data is a numpy array
import pandas as pd
train_images = pd.read_pickle(cwd + '/data/train_images.pkl')
train_labels = pd.read_csv(cwd +'/data/train_labels.csv')
test_images = pd.read_pickle(cwd + '/data/test_images.pkl')

print(train_images.shape)
print(train_labels.shape)

import matplotlib.pyplot as plt

# show image with id 13
img_idx = 13

plt.title('Label: {}'.format(train_labels.iloc[img_idx]['Category']))
plt.imshow(train_images[img_idx])
plt.show()

#############

### here needs to add crop train/test image by largest bounding box 
### dont know how to do this.
### give up.

#############

'''
Step 2: Train models
'''
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
import tensorflow as tf

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torchvision import models, transforms
from datetime import datetime
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import cv2

# x_tr = np.expand_dims(train_images, axis=1)
# x_ts = np.expand_dims(test_images, axis=1)
# y_tr = train_labels.values[:, 1]
#x_ts = np.array(test_images)

x_tr = np.array(train_images)
y_tr = np.array(train_labels)
x_ts = np.expand_dims(test_images, axis=1)

print(x_tr.shape)
print(x_ts.shape)


from keras.utils import np_utils
#X_train, x_ts = x_tr/255, x_ts/255
#X_test = np.expand_dims(X_test, axis=0)
X_train = x_tr.reshape(x_tr.shape[0], 64, 64,1)

X_test = x_ts.reshape(x_ts.shape[0], 64, 64, 1)
#y_train = to_categorical(y_tr)
y_train = y_tr
# y_train = to_categorical(y_tr['Category'], num_classes=10, dtype='float32')

# x_train = X_train.astype('float32')
# x_test = x_ts.astype('float32')
# x_train /= 255
# x_test /= 255

X_train, X_vali, y_train, y_vali = train_test_split(X_train,
                                                    y_train,
                                                    test_size=0.2,
                                                    random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)

model = Sequential([
    Conv2D(32, (5, 5), padding='Same', activation='relu', input_shape=(64, 64, 1)),
    Conv2D(32, (5, 5), padding='Same', activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), padding='Same', activation='relu'),
    Conv2D(64, (3, 3), padding='Same', activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(500, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(lr=0.001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                              factor=0.5,
                              patience=3,
                              verbose=1,
                              min_lr=0.1)
                              # min_lr=0.00001)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.3)

#batch size = 128 first
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),
                              epochs=5,
                              validation_data=datagen.flow(X_vali, y_vali, batch_size=128),
                              validation_steps=20,
                              verbose=1,
                              steps_per_epoch=X_train.shape[0] // 64,
                              callbacks=[reduce_lr])

predictions = model.predict_classes(X_test, verbose=0)

pd.DataFrame(predictions).to_csv("mnist_prediction.csv")
