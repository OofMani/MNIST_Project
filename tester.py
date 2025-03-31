import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

'''LOADING THE MODEL'''
model = tf.keras.models.load_model('mnist.keras') # Load the model
model.summary() # Print the model summary

'''LOADING THE DATASET'''
mnist = tf.keras.datasets.mnist # Load the MNIST dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data() # Load the training and test data

x_train = tf.keras.utils.normalize(x_train, axis=1) # Normalize the training data
x_test = tf.keras.utils.normalize(x_test, axis=1) # Normalize the test data

'''TESTING THE MODEL'''
loss, accuracy = model.evaluate(x_test, y_test) # Evaluate the model on the test data
print ('\n' + '='  * 50 + '\n')
print('Test loss:', loss) # Print the test loss
print('Test accuracy:', accuracy) # Print the test accuracy
print ('\n' + '='  * 50 + '\n')