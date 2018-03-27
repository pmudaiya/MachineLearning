# Artificial Neural Network
"""
Best link to follow to install all 3
https://www.udemy.com/machinelearning/learn/v4/questions/2320940

Dataset for code is at
www.superdatascience.com/machine-learning
"""
# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

#important Points keep CUDA version 9.0 because 9.1 is not supported
# cudnn should be 7.0 7.1 is not supported

# Installing Keras
# pip install --upgrade keras

#Part 1 Building CNN

#Importing Libraries
from keras.models import Sequential
#to Intialise CNN

from keras.layers import Conv2D
#Add convolution layer

from keras.layers import MaxPooling2D
#Maxpooling layer

from keras.layers import Flatten
#Flattening of images to import in CNN

from keras.layers import Dense
#Intiase Fully Connected layer like ANN

#Creating or Intialising CNN
classifier=Sequential()

#Adding COnvolution Layer
classifier.add(Conv2D(32,(3,3),activation="relu",input_shape=(64,64,3)))

#Adding MaxPooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#adding second covolution layer
classifier.add(Conv2D(32,(3,3),activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))


#flattening mages into vector
classifier.add(Flatten())

#Adding Hidden_Layer
classifier.add(Dense(128,activation='relu'))

#Adding output Layer
classifier.add(Dense(1,activation='sigmoid'))

#Compiling CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

"""
Main Aim is Image AUgmentation:- Creating more images from our datset so that our
model can find correlation easily
8000 images in traing set might not to easy
and plus it reduces Overfitting
"""

#Generates Random Images from our Dataset by rescaling,zooming etc
#so what when more images are egnerated no image looks same
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Creates Training Set from data ins pecified Location
#target size is input image size
#class mode tells whether output is binary(2 Classes) or more binary means two
#batch_size tells size of batch of images that is passed throug cnn
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

#Generates Test Set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 50,
                         validation_data = test_set,
                         nb_val_samples = 2000)













