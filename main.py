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

'''
Step 2: Train models
'''

