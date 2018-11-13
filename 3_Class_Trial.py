# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

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

# Part 2 - Fitting the CNN to the images
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/home/astar/Desktop/DL_Images/training_set',
target_size = (160, 60),
classes=['FORWARD', 'LEFT', 'STOP'],
batch_size = 32,
class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('/home/astar/Desktop/DL_Images/test_set',
target_size = (160, 60),
classes=['FORWARD', 'LEFT', 'STOP'],
batch_size = 32,
class_mode = 'categorical')

classifier.fit_generator(training_set,
steps_per_epoch = 10,
epochs = 1,
validation_data = test_set,
validation_steps = 10)

# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/home/astar/Desktop/DL_Images/validation_set/STOP/stop_514.jpg', target_size = (160, 60))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print (result)
print (result[0][0])
#training_set.class_indices

if result[0][0] == 1:
    prediction = 'forward'
elif result[0][1] == 1:
    prediction = 'left'
else:
    prediction = 'stop'
print (prediction)

# serialize model to JSON
model_json = classifier.to_json(indent=4)
with open("/home/astar/Desktop/DL_Images/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("/home/astar/Desktop/DL_Images/model.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('/home/astar/Desktop/DL_Images/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/home/astar/Desktop/DL_Images/model.h5")
print("Loaded model from disk")

"""evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))"""

