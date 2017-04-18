import csv
import cv2
import numpy as np

# read lines from training data csv file
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# capture images from training data left, right and center cameras and their associated steering angles        
images = []  
measurements = []
for line in lines:
    # set correction parameter for left and right cameras
    correction = 0.2 
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

# flip images from left, right and center cameras and inverse corresponding steering angles to augment training dataset and remove bias
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
        
X_train = np.array(augmented_images) 
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# build model architecture
model = Sequential()
# normalize input images' pixel values
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
# crop top (sky,trees) and bottom (car hood) pixels of images to remove noise
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
 
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, batch_size=64, nb_epoch=2, validation_split=0.2, shuffle=True)

# save trained model
model.save('model.h5')
exit()