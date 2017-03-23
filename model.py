# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:18:00 2017

@author: lochappy
"""

import csv, cv2, h5py
import numpy as np

dataFolder = './sample_data'
with open(dataFolder + '/driving_log.csv') as f:
    reader = csv.reader(f)
    next(reader) #this skip the header
    dataAnn = [line for line in reader]
    
from random import shuffle
shuffle(dataAnn)

center_images = np.array([cv2.imread(dataFolder+'/'+line[0]) for line in dataAnn],dtype=np.float32)
steering_angles = np.array([float(line[3]) for line in dataAnn],dtype=np.float32)

flipped_center_images = np.array([np.fliplr(img)for img in center_images],dtype=np.float32)
flipped_steering_angles = -steering_angles

#import matplotlib.pyplot as plt
#plt.figure()
#plt.imshow(center_images[0])
#plt.figure()
#plt.imshow(flipped_center_images[0])

X_train = np.append(center_images,flipped_center_images,axis=0)
Y_train = np.append(steering_angles,flipped_steering_angles,axis=0)

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Lambda, Activation, pooling, PReLU

model = Sequential()
model.add(Lambda(lambda x: (x/255.0 - 0.5),input_shape=(160,320,3)))

model.add(Conv2D(6,5,5))
model.add(PReLU())
model.add(pooling.MaxPool2D())

model.add(Conv2D(6,5,5))
model.add(PReLU())
model.add(pooling.MaxPool2D())

model.add(Flatten())
model.add(Dense(120))
#model.add(PReLU())
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

'''
saves the model weights after each epoch if the validation loss decreased
'''
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath="model.h5", verbose=1, save_best_only=True)

model.fit(x=X_train,y=Y_train,validation_split=0.2,shuffle=True,nb_epoch=5,callbacks=[checkpointer])

#model.save('behavior_cloning_model.h5')

