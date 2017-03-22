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

center_images = np.array([cv2.imread(dataFolder+'/'+line[0]) for line in dataAnn],dtype=np.float32)
steering_angles = np.array([float(line[3]) for line in dataAnn],dtype=np.float32)

X_train = center_images
Y_train = steering_angles

from keras.models import Sequential
from keras.layers import Deconv2D, Dense, Flatten, Lambda

model = Sequential()
model.add(Lambda(lambda x: (x/255.0 - 0.5),input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(x=X_train,y=Y_train,validation_split=0.2,shuffle=True,nb_epoch=7)

model.save('behavior_cloning_model.h5')