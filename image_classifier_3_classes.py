import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from keras.models import model_from_json

# To plot image along with one hot encoded label for verification of data set
# This works on Jupyter notebook, for PyCharm, not sure how to use

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

# Set directories for image flow

train_path = '/home/astar/Desktop/DL_Images/training_set'
test_path = '/home/astar/Desktop/DL_Images/test_set'
valid_path = '/home/astar/Desktop/DL_Images/validation_set'

# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (160, 60, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
valid_datagen = ImageDataGenerator(rescale = 1./255)

# Start image flow from directory and set classes for one-hot encoding
# class_mode is set to categorical as default so we do not have to mention it here

train_batches = train_datagen.flow_from_directory(train_path, target_size =(160, 60),
                                                  classes=['FORWARD', 'LEFT', 'STOP'], batch_size=10)
test_batches = test_datagen.flow_from_directory(test_path, target_size =(160, 60),
                                                classes=['FORWARD', 'LEFT', 'STOP'], batch_size=10)
valid_batches = valid_datagen.flow_from_directory(valid_path, target_size =(160, 60),
                                                  classes=['FORWARD', 'LEFT', 'STOP'], batch_size=10)

# Code snippet to print one hot encoded labels

"""imgs, labels = next(train_batches)
print(labels)"""

classifier.fit_generator(train_batches, steps_per_epoch = 10, epochs = 1, validation_data = test_batches, validation_steps = 10)





