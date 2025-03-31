import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

'''LOADING THE DATASET'''
mnist = tf.keras.datasets.mnist # Load the MNIST dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data() # Load the training and test data

x_train = tf.keras.utils.normalize(x_train, axis=1) # Normalize the training data
x_test = tf.keras.utils.normalize(x_test, axis=1) # Normalize the test data

'''CREATING THE MODEL'''
model = tf.keras.models.Sequential() # Create a sequential model
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) #Add a flatten layer to convert the 2D images into 1D arrays
model.add(tf.keras.layers.Dense(128, activation='relu')) # Add a dense layer with 128 neurons and RELU activation function
model.add(tf.keras.layers.Dense(128, activation='relu')) # Add another dense layer with 128 neurons and RELU activation function
model.add(tf.keras.layers.Dense(10, activation='softmax')) # Add a dense layer with 10 neurons and softmax activation function

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy']) # Compile the model with Adam optimizer and sparse categorical crossentropy loss function

'''TRAINING THE MODEL'''
model.fit(x_train, y_train, epochs = 3) # Train the model for 3 epochs
model.save('mnist.keras') # Save the model