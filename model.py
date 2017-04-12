import csv
import cv2
import numpy as np

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []  
measurements = []
for line in lines:
    correction = 0.2 # this is a parameter to tune
    steering_center = float(line[3])
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = 'data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        if i == 0:
            measurement = steering_center
        elif i == 1:
            measurement = steering_center + correction
        else:
            measurement = steering_center - correction
        measurements.append(measurement)
        
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
        
X_train = np.array(augmented_images) 
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(16,5,5,activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dense(1))
 
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, batch_size=64, nb_epoch=3, validation_split=0.2, shuffle=True)

model.save('model.h5')
exit()