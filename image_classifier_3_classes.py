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
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.metrics import confusion_matrix
import itertools
from keras.models import model_from_json

# To plot image along with one hot encoded label for verification of data set

def plots1(ims, figsize=(16,6), rows=2, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        #print(len(ims))
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])
    plt.show()

# Set directories for image flow

train_path = '/home/astar/Desktop/DL_Stuff/DL_Final/training_set'
test_path = '/home/astar/Desktop/DL_Stuff/DL_Final/test_set'
valid_path = '/home/astar/Desktop/DL_Stuff/DL_Final/validation_set'

classifier = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(160, 427, 3)),
    Flatten(),
    Dense(4, activation='softmax'),
])

classifier.compile(Adam(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
valid_datagen = ImageDataGenerator(rescale = 1./255)

# Start image flow from directory and set classes for one-hot encoding
# class_mode is set to categorical as default so we do not have to mention it here

train_batches = train_datagen.flow_from_directory(train_path, target_size =(160, 427),
                                                  classes=['FORWARD', 'LEFT', 'RIGHT', 'STOP'], batch_size=10)
test_batches = test_datagen.flow_from_directory(test_path, target_size =(160, 427),
                                                classes=['FORWARD', 'LEFT', 'RIGHT', 'STOP'], batch_size=10)
valid_batches = valid_datagen.flow_from_directory(valid_path, target_size =(160, 427),
                                                  classes=['FORWARD', 'LEFT', 'RIGHT', 'STOP'], batch_size=10)

# Code snippet to print one hot encoded labels

imgs, labels = next(train_batches)
#plots1(imgs, titles=labels)

classifier.fit_generator(train_batches, steps_per_epoch = 186, epochs = 25, validation_data = test_batches, validation_steps = 48, verbose=1)

model_path='/home/astar/Desktop/DL_Stuff/DL_Final/my_model.h5'
classifier.save(model_path)  # creates a HDF5 file 'my_model.h5'

#Test with new image

"""test_image = image.load_img('/home/astar/Desktop/DL_Images_New/validation_set/STOP/stop_345.jpg', target_size = (120, 320))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print (result)
train_batches.class_indices
if result[0][0] == 1:
    prediction = 'forward'
elif result[0][1] == 1:
    prediction = 'left'
else:
    prediction = 'stop'
print (prediction)"""

model_path='/home/astar/Desktop/DL_Stuff/DL_Final/my_model.h5'
model = load_model(model_path)
test_image = image.load_img('/home/astar/Desktop/DL_Stuff/DL_Final/validation_set/FORWARD/forward_536.jpg', target_size = (160, 427))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print (result)
train_batches.class_indices
if result[0][0] == 1:
    prediction = 'forward'
elif result[0][1] == 1:
    prediction = 'left'
elif result[0][2] == 1:
    prediction = 'right'
else:
    prediction = 'stop'
print (prediction)

