import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils

from keras.layers.convolutional import *
from keras.utils import conv_utils
from keras.layers.core import *
from keras.engine.topology import Layer

import numpy as np

import scipy.io as sio
import random
from random import shuffle
import matplotlib.pyplot as plt

import os.path
import errno


import scipy.ndimage
from skimage.transform import rotate

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import dill as pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.layers import concatenate
def loadIndianPinesData():
    data = sio.loadmat(os.path.join( 'Indian_pines.mat'))['indian_pines']
    labels = sio.loadmat(os.path.join('Indian_pines_gt.mat'))['indian_pines_gt']
    return data, labels
  
def splitTrainTestSet(X, y, testRatio=0.10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=345,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test
  
def standartizeData(X):
    newX = np.reshape(X, (-1, X.shape[2]))
    scaler = preprocessing.StandardScaler().fit(newX)  
    newX = scaler.transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1],X.shape[2]))
    return newX, scaler
  



def applyPCA(X, numComponents=75, drawPlot = False):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    
    return newX, pca
  def createPatches(X, y, windowSize=3):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1    
    patchesData = patchesData[patchesLabels>0,:,:,:]
    patchesLabels = patchesLabels[patchesLabels>0]
    patchesLabels -= 1
    return patchesData, patchesLabels

def Patch(data,height_index,width_index):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    return patch
  
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX
    n_classes=16

def get_batch(n,X_data,y_data):
        n_examples,w,h,ch = X_data.shape
        """Create batch of n pairs, half same class, half different class"""
        categories = rng.choice(n_classes,size=(n,),replace=False)
        pairs=[np.zeros((n, h, w,200)) for i in range(2)]
        targets=np.zeros((n,))
        targets[n//2:] = 1
        for i in range(n):
            category = categories[i]
            idx_1 = rng.randint(0,n_examples)
            #print(pairs[0][0].shape, X_train[0].shape,np.where(y_train == category)[0][i])
            pairs[0][i] = X_data[np.where(y_data == category)[0][i]]
            idx_2 = rng.randint(0,n_examples)
            #pick images of same class for 1st half, different for 2nd
            category_2 = category if i >= n//2 else (category + rng.randint(1,n_classes)) % n_classes
            pairs[1][i] = X_data[np.where(y_data == category_2)[0][i]]
        return pairs, targets

#(inputs,targets)=get_batch(15,X_train,y_train)



def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)
#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)
def getModel(input_shape):
  
    input_shape=(9,9,200)
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    model = Sequential()
    model.add(Conv2D(100, (3, 3), padding='same', input_shape=input_shape,data_format='channels_last'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(5, 5), strides=None, padding='same'))
    
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1000,kernel_initializer=W_init,bias_initializer=b_init))
    model.add(Activation('relu'))
         
    model.add(Dense(500))
    model.add(Activation('relu'))
            
    model.add(Dense(300))
    model.add(Activation('relu'))
    
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    #merge two encoded inputs with the l1 distance between them
    L1_distance = lambda x: K.abs(x[0]-x[1])
    both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
    prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(both)
    siamese_net = Model(input=[left_input,right_input],output=prediction)
    
    return siamese_net,model
    windowSize=9
X,y= loadIndianPinesData()
print(X.shape,y.shape)
numComponents=200
isPCA=True
# PCA
if isPCA == True:
    X,pca = applyPCA(X,numComponents=numComponents)
print(X.shape,y.shape)
XPatches, yPatches = createPatches(X, y, windowSize=windowSize)

X_train, X_test, y_train, y_test = splitTrainTestSet(XPatches, yPatches, 0.25)


print(X_train.shape)
print(y_test.shape)


#X_test  = np.reshape(X_test, (X_test.shape[0], X_test.shape[3], X_test.shape[1], X_test.shape[2]))
y_test_ct = np_utils.to_categorical(y_test)
#X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[3], X_train.shape[1], X_train.shape[2]))
y_train_ct = np_utils.to_categorical(y_train)
print(y_train.shape)
print(X_train.shape)
print(X_train[0].shape)
print(y_train[0])

model, dum = getModel(X_train[0].shape)
opt = keras.optimizers.Adam(lr=0.0001,decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  for i in range(400):
    (inputs,targets)=get_batch(15,X_train,y_train)
    loss=model.train_on_batch(inputs,targets)
  print(loss)
  model.summary()
  dum.summary()
