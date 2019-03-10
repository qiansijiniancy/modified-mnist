### Mini Project 3
### McGill University
### author: Group 16

'''
Step 1: Load Data
'''

import numpy as np
import pandas as pd

import os
cwd = os.getcwd()
input_path = cwd+"/data"
print(os.listdir(input_path))

## load data
train_images = pd.read_pickle(cwd + '/data/train_images.pkl')
#train_labels = pd.read_csv(cwd +'/data/train_labels.csv')
test_images = pd.read_pickle(cwd + '/data/test_images.pkl')
train_labels = pd.read_csv(cwd +'/data/train_labels.csv').values[:, 1]

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(train_labels[1:20])


#Let's show image with id 16
# import matplotlib.pyplot as plt
# img_idx = 13
#
# plt.title('Label: {}'.format(train_labels.iloc[img_idx]['Category']))
# plt.imshow(train_images[img_idx])
# plt.show()

'''
Step 2: Train models
'''
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import cv2

x_tr = np.array(train_images)
#y_tr = np.array(train_labels)
x_ts = np.expand_dims(test_images, axis=1)

print(x_tr.shape)
print(x_ts.shape)

X_train = x_tr.reshape(x_tr.shape[0], 64, 64,1)
X_test = x_ts.reshape(x_ts.shape[0], 64, 64, 1)


from keras.utils import np_utils
y_tr = np_utils.to_categorical(train_labels)

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

num_classes = 10
import keras
import keras.utils
from keras import utils as np_utils


X_train, X_vali, y_train, y_vali = train_test_split(X_train,
                                                    y_tr,
                                                    test_size=0.2,
                                                    random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)

print(y_train[0:20])

import keras
input_shape = (64,64,1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

epochs = 1
batch_size = 128
# Fit the model weights.
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_vali,y_vali))

predictions = model.predict_classes(X_test)
print(predictions[0:20])

pd.DataFrame(predictions).to_csv("mnist_prediction.csv")
