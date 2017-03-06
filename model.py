#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:50:26 2017

@author: ahmedshehata
"""

import cv2
import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

#Reading the training data from the data file.
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#Split the training data into validation(20%) and training data(80%).
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Using generator to load data and preprocess it on the fly.
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            correction = 0.2
            images = []
            angles = []
            for batch_sample in batch_samples:
                name1 = './IMG/'+batch_sample[0].split('/')[-1]
                name2 = './IMG/'+batch_sample[1].split('/')[-1]
                name3 = './IMG/'+batch_sample[2].split('/')[-1]
                center_image = cv2.imread(name1)
                center_angle = float(batch_sample[3])
                left_image = cv2.imread(name2)
                left_angle = center_angle + correction
                right_image = cv2.imread(name3)
                right_angle = center_angle - correction
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)
            
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, angles):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement*-1.0)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            #X_train = X_train.reshape(X_train.shape[0], 320, 160, 3)
            yield shuffle(X_train, y_train)

#Training the data using the generator function.
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()

#Preprocessing the data.
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(320, 160, 3), output_shape=(320, 160, 3)))

#Cropping unwanted parts of the image (sky, hills and car hood).
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

#Using NVIDIA architecture for training the data.
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation = 'relu'))

#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))

#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#compiling the data using Adam optimizer and mean square error loss.
model.compile(loss='mse', optimizer='adam')
history_object=model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

#Saving the model.
model.save('model.h5')

#print the keys contained in the history object.
print(history_object.history.keys())

#plot the training and validation loss for each epoch.
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
