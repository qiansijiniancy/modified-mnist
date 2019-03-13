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
import sys
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import exposure

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.model_selection import train_test_split

# speed-up using multithreads
cv2.setUseOptimized(True);
cv2.setNumThreads(4);


cwd = os.getcwd()
input_path = cwd+"/data"
print(os.listdir(input_path))

# Any results you write to the current directory are saved as output.

#Files are stored in pickle format.
#Load them like how you load any pickle. The data is a numpy array
import pandas as pd
# train_images = pd.read_pickle(cwd + '/data/train_images.pkl')
# train_labels = pd.read_csv(cwd +'/data/train_labels.csv')
# test_images = pd.read_pickle(cwd + '/data/test_images.pkl')

# print(train_images.shape)
# print(train_labels.shape)

# import matplotlib.pyplot as plt

# # show image with id 13
# img_idx = 13

# plt.title('Label: {}'.format(train_labels.iloc[img_idx]['Category']))
# plt.imshow(train_images[img_idx])
# plt.show()

X_data = pd.read_pickle('data/train_images.pkl')
y_data = pd.read_csv('data/train_labels.csv')
X_test = pd.read_pickle('data/test_images.pkl')

'''
Step 2: Preprocess Image and Crop largest bounding box digit
'''


def remove_background_noise(im):
    # stretch contrast to have only binary valued pixels
    processed = exposure.rescale_intensity(im, in_range=(1, 0))
    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

    return processed


def get_bound_regions(im, method='f'):
    # see https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/

    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # set input image on which we will run segmentation
    ss.setBaseImage(im)

    # Switch to fast but low recall Selective Search method
    if (method == 'f'):
        ss.switchToSelectiveSearchFast()

    # Switch to high recall but slow Selective Search method
    else:
        ss.switchToSelectiveSearchQuality()

    # run selective search segmentation on input image
    rects = ss.process()

    return rects


def get_best_regions(rects, h, w):
    # To improve
    best_rect = None
    max_bound_area = 0
    limit = h * w

    for rect in rects:
        bound_area = rect[2] * rect[3]
        if bound_area > max_bound_area and bound_area < limit:
            max_bound_area = rect[2] * rect[3]
            best_rect = rect

    return [best_rect]


def get_best_regions2(rects, h, w):
    length = len(rects)
    return rects[0:min(length, 3)]

def display_image_with_bound(im, procesed, bound_regions, cropped):
    # create a copy of original image
    imOut = im.copy()

    for i, bound_region in enumerate(bound_regions):
        x, y, w, h = bound_region
        cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)

#     # show output
#     plt.subplot(141),plt.imshow(im), plt.title('input')
#     plt.subplot(142),plt.imshow(procesed), plt.title('processed')
#     plt.subplot(143),plt.imshow(imOut), plt.title('result')
#     plt.subplot(144),plt.imshow(cropped), plt.title('cropped/resized')
#     plt.show()


# This method calls the above three and returns a (resized) cropped image
def find_best_bound_region(im_orig, display=False):
    im_processed = remove_background_noise(im_orig / 255.0)
    bound_regions = get_bound_regions(im_processed)
    best_regions = get_best_regions(bound_regions, im_processed.shape[0], im_processed.shape[1])
    x, y, w, h = best_regions[0][0], best_regions[0][1], best_regions[0][2], best_regions[0][3]
    im_cropped = im_processed[y:y + h, x:x + w]
    image_resized = resize(im_cropped, (im_orig.shape[0], im_orig.shape[1]))
    if display:
        display_image_with_bound(im_orig, im_processed, best_regions, image_resized)
    return image_resized


#visualize results
j=0
for i in np.random.randint(0, len(X_data), size=(20,)):
    print(j, y_data['Category'].iloc[i])
    find_best_bound_region(X_data[i], display=True)
    j+=1


def pre_process_images(X_data):
    X_data_processed = []

    for i in range(len(X_data)):
        X_data_processed.append(find_best_bound_region(X_data[i]))
        if i % 5000 == 0:
            print(i)

    return np.array(X_data_processed)


X_data_processed = pre_process_images(X_data)

plt.imshow(X_data_processed[2])


X_data_processed2 = X_data_processed[:, :, :, 0].reshape(X_data_processed.shape[0], 64, 64, 1)
y_data = keras.utils.to_categorical(y_data['Category'])

X_data_processed2.shape

'''
Step 3:
Train Model

'''
X_train, X_val, y_train, y_val = train_test_split(X_data_processed2, y_data, test_size=0.2, random_state=39)

num_classes = 10
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam, Adadelta

import keras
input_shape = (64,64,1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print(model.summary())

'''
This part is under testing.
data_generator = ImageDataGenerator(rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2)

model.fit_generator(data_generator.flow(X_train, y_train,batch_size=batch_size),
          epochs=epochs,
          verbose=1,
          steps_per_epoch=num_steps,
          validation_data=(X_val,y_val))
'''

epochs = 10
batch_size = 128
num_steps= 1000

model.fit_generator((X_train, y_train),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
   #       steps_per_epoch=num_steps,
          validation_data=(X_val,y_val))


X_test_processed = pre_process_images(X_test)
X_test_processed2 = X_test_processed[:,:,:,0].reshape(X_test_processed.shape[0], 64, 64, 1)
print(X_test_processed2.shape)

predictions = model.predict_classes(X_test_processed2, verbose=2)
pd.DataFrame(predictions).to_csv("mar_12_mnist_prediction.csv")


