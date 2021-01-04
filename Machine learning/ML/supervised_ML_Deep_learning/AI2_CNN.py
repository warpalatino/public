import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ---------------------------
# deep learning - CNN for classification

# ***
# we have a big dataset of pictures (250mb) with cats and dogs, locally stored (in folder 5 - Tech)
# ***


# [1] load and pre-process data
# ------
# -- we apply image augmentation/transformations here to avoid over-fitting => we apply shifts and rotations and flips and zooms to the images
# https://keras.io/api/preprocessing/image/
# rescale property is about feature scaling while other properties below are from a Keras example, click link above
# the model will look at images in batches as usual
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
training_set = train_datagen.flow_from_directory('../data/CNN pics dataset/training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255) # we scale but do not transform/augment the testset images as we need the originals to compare the effectiveness of our training/learning
test_set = test_datagen.flow_from_directory('../data/CNN pics dataset/training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')




# [2] define the model, according to new keras instructions 
# https://www.tensorflow.org/guide/keras/rnn
# ------
model = keras.Sequential()
# -- first convolution and pooling
model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3])) # filters are output filters in convolution, kernel is the CNN feature detector square, input shape for first input layer
model.add(layers.MaxPool2D(pool_size=2, strides=2)) # size of the pool (or set of pixels) to squeeze into one pixel in feature map, while strides is about shifting the frame of pixels to capture next pixels to observe
# -- second convolution and pooling
model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(layers.MaxPool2D(pool_size=2, strides=2))
# -- flatten 
model.add(layers.Flatten()) # here we take all the pixels and flatten them into a vector that keeps the dimensional charateristics of a picture
# -- connect
model.add(layers.Dense(units=128, activation='relu'))   # neurons are high here because processing images is more complex and we may get more accuracy
# -- output layer
model.add(layers.Dense(units=1, activation='sigmoid'))  # we need just one neuron for binary classification as output (0/1, or cat/dog)




# [3] fit (and run/train) the model 
# ------
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy']) # adam is a way to perform the stochastic gradient descent
model.fit(x = training_set, validation_data = test_set, epochs = 25)    
# print(model.summary())




# [4] try a first prediction around a single picture
# ------
# -- load a specific image to observe after training and predict
test_image = image.load_img('../data/CNN pics dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
# -- convert the image into a numpy array, then expand the array into an extra dimension as images will be processed in batches (batch => extra dimension)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
# -- make a prediction in terms of either 0 or 1
result = cnn.predict(test_image)
# -- decode: if prediction is 1, then dog; if 0, then cat; we know what index corresponds to which class by calling the attribute class_indices as below...
print(training_set.class_indices)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)


