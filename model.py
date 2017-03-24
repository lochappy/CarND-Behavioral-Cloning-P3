# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:18:00 2017

@author: lochappy
"""

import csv, cv2, h5py, sklearn
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Lambda, Activation, pooling, PReLU, Cropping2D, Dropout

def LeNet():
    ''' Return lenet architecture '''
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0 - 0.5),input_shape=(160,320,3)))
    
    model.add(Cropping2D(cropping=((50,25),(0,0))))
    
    model.add(Conv2D(24,5,5))
    model.add(PReLU())
    model.add(pooling.MaxPool2D())
    
    model.add(Conv2D(48,3,3))
    model.add(PReLU())
    model.add(pooling.MaxPool2D())
    
    model.add(Conv2D(96,3,3))
    model.add(PReLU())
    model.add(pooling.MaxPool2D())
    
    model.add(Flatten())
    model.add(Dense(256))
    #model.add(PReLU())
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1))
    
    return model
    
def NvidiaDriving():
    ''' Return NvidiaDriving architecture '''
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0 - 0.5),input_shape=(160,320,3)))
    
    model.add(Cropping2D(cropping=((50,25),(0,0))))
    
    model.add(Conv2D(24,5,5))#, subsample=(2,2)))
    #model.add(Activation('relu'))
    model.add(PReLU())
    model.add(pooling.MaxPool2D())
    
    model.add(Conv2D(36,3,3))#, subsample=(2,2)))
    #model.add(Activation('relu'))
    model.add(PReLU())
    model.add(pooling.MaxPool2D())
    
    model.add(Conv2D(48,3,3))#, subsample=(2,2)))
    #model.add(Activation('relu'))
    model.add(PReLU())
    model.add(pooling.MaxPool2D())
    
    model.add(Conv2D(64,3,3))
    #model.add(Activation('relu'))
    model.add(PReLU())
    
    model.add(Conv2D(64,3,3))
    #model.add(Activation('relu'))
    model.add(PReLU())
    
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))
    
    return model
    

dataFolder = './my_data'
with open(dataFolder + '/driving_log.csv') as f:
    reader = csv.reader(f)
    next(reader) #this skip the header
    dataAnn = [line for line in reader]
    
from random import shuffle
shuffle(dataAnn)


center_images = np.array([cv2.imread(line[0]) for line in dataAnn],dtype=np.float32)
steering_angles = np.array([float(line[3]) for line in dataAnn],dtype=np.float32)

flipped_center_images = np.array([np.fliplr(img)for img in center_images],dtype=np.float32)
flipped_steering_angles = -steering_angles

X_train = np.append(center_images,flipped_center_images,axis=0)
Y_train = np.append(steering_angles,flipped_steering_angles,axis=0)

model = NvidiaDriving()

from keras import optimizers
adam = optimizers.Adam(lr=0.0001)

model.compile(loss='mse',optimizer=adam)

'''
saves the model weights after each epoch if the validation loss decreased
'''
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath="model.h5", verbose=1, save_best_only=True)

model.fit(x=X_train,y=Y_train,validation_split=0.2,shuffle=True,nb_epoch=15,callbacks=[checkpointer],batch_size=64)

#from sklearn.model_selection import train_test_split
#train_samples, validation_samples = train_test_split(dataAnn, test_size=0.2)
#
#def generator(samples, batch_size=32):
#    num_samples = len(samples)
#    while 1: # Loop forever so the generator never terminates
#        shuffle(samples)
#        for offset in range(0, num_samples, batch_size):
#            batch_samples = samples[offset:offset+batch_size]
#
#            center_images = np.array([cv2.imread(line[0]) for line in batch_samples],dtype=np.float32)
#            steering_angles = np.array([float(line[3]) for line in batch_samples],dtype=np.float32)
#            
#            flipped_center_images = np.array([np.fliplr(img)for img in center_images],dtype=np.float32)
#            flipped_steering_angles = -steering_angles
#            
#            X_train = np.append(center_images,flipped_center_images,axis=0)
#            Y_train = np.append(steering_angles,flipped_steering_angles,axis=0)
#
#            yield sklearn.utils.shuffle(X_train, Y_train)
#
#
## compile and train the model using the generator function
#train_generator = generator(train_samples, batch_size=32)
#validation_generator = generator(validation_samples, batch_size=32
#
#model = NvidiaDriving()
#
#from keras import optimizers
#adam = optimizers.Adam(lr=0.0001)
#
#model.compile(loss='mse',optimizer=adam)
#
#'''
#saves the model weights after each epoch if the validation loss decreased
#'''
#from keras.callbacks import ModelCheckpoint
#checkpointer = ModelCheckpoint(filepath="model.h5", verbose=1, save_best_only=True)
#
#model.fit_generator(train_generator, samples_per_epoch= len(train_samples), \
#                    validation_data=validation_generator, nb_val_samples=len(validation_samples), \
#                    nb_epoch=15,callbacks=[checkpointer])

#model.save('behavior_cloning_model.h5')

