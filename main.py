###  Mini project 3
### McGill University
### Group 16

'''
Step 1: Load Data
'''
import os

# ref: https://www.kaggle.com/comp551f2018ta/example-to-load-data
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.backends.cudnn
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torchvision import models, transforms
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch.optim as optim

#### DATA LOADING AS ARRAY#####################
cwd = os.getcwd()
input_path = cwd+"/data"
print(os.listdir(input_path))

## load data
train_images = pd.read_pickle(cwd + '/data/train_images.pkl')
test_images = pd.read_pickle(cwd + '/data/test_images.pkl')
train_labels = pd.read_csv(cwd +'/data/train_labels.csv').values[:, 1]

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(train_labels[1:20])
###################################################


##############TRAIN DATA#########################
x_tr = np.array(train_images)
#y_tr = np.array(train_labels)
x_ts = np.expand_dims(test_images, axis=1)

print(x_tr.shape)
print(x_ts.shape)

X_train = x_tr.reshape(x_tr.shape[0], 64, 64,1)
X_test = x_ts.reshape(x_ts.shape[0], 64, 64, 1)

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
from keras.utils import np_utils
y_tr = np_utils.to_categorical(train_labels) # should be changed(not keras)

X_train, X_vali, y_train, y_vali = train_test_split(X_train,
                                                    y_tr,
                                                    test_size=0.2,
                                                    random_state=42)



class TwoLayerNet(torch.nn.Module):
    def __init__(self):
        super(TwoLayerNet, self).__init__() #####??????????
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = TwoLayerNet()

# loss function
criterion = torch.nn.MSELoss(reduction='sum')
# optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-4)



losses = []

for epoch in range(50):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    losses.append(loss.data.item())
    print(f"Epoch : {epoch}    Loss : {loss.data.item()}")

    # Reset gradients to zero, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()






#
# class TwoLayerNet(torch.nn.Module):
#     def __init__(self, D_in, H, D_out):
#         """
#         In the constructor we instantiate two nn.Linear modules and assign them as
#         member variables.
#
#         Args:
#             - D_in : input dimension of the data
#             - H : size of the first hidden layer
#             - D_out : size of the output/ second layer
#         """
#         super(TwoLayerNet, self).__init__()  # intialize recursively
#         self.linear1 = torch.nn.Linear(D_in, H)  # create a linear layer
#         self.linear2 = torch.nn.Linear(H, D_out)
#
#     def forward(self, x):
#         """
#         In the forward function we accept a Tensor of input data and
#         return a tensor of output data. We can use
#         Modules defined in the constructor as well as arbitrary
#         operators on Variables.
#         """
#         h_relu = self.linear1(x)
#         y_pred = self.linear2(h_relu)
#         return y_pred
#
#
