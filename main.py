# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:13:57 2023

@author: mathe
"""

# Import necessary libraries
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard

#Defining material parameters
mu = 0.5
lmda = 1000

# Load data from files
input_feature = np.array(pd.read_csv('Ux.txt', delimiter = "\t"))[:,1:4]
ux = np.array(pd.read_csv('Ux.txt', delimiter = "\t"))[:,4]
uy = np.array(pd.read_csv('Uy.txt', delimiter = "\t"))[:,4]
uz = np.zeros((uy.shape[0],1))
displacement = np.concatenate((ux[:,np.newaxis], uy[:,np.newaxis], uz), axis=1)
bc_data = np.array(pd.read_csv('Ux.txt', delimiter = "\t"))
bc_data = np.concatenate((bc_data, uy[:,np.newaxis]), axis=1)

# Create a boolean index that is True for rows where the value in the 3rd column is 10 
index = (bc_data[:, 2] == 10)

#BCs is a 3 column array (x,y,z) vectors of displacement BCs or could just be a one column?
boundary_conditions = np.zeros((bc_data.shape[0],1))

# Set the values in to 10 when y = 10
boundary_conditions[index] = 10 

# Create a boolean index that is True for rows where the value in the 3rd column is 0 or 10
index = (bc_data[:, 2] == 0) | (bc_data[:, 2] == 10)

# Create a new column with all values set to 0
bc_present = np.zeros((bc_data.shape[0], 1))

# Set the values in the new column to 1 for rows where the index is True
bc_present[index] = 1

## Concatenate the old array and the new column to create the new array
boundary_conditions = np.concatenate((bc_present, boundary_conditions), axis=1)

# Creating the input feature vector
input_feature = np.concatenate((input_feature, boundary_conditions), axis=1)

# Normalize the data
scaler = MinMaxScaler()
input_feature = scaler.fit_transform(input_feature)
displacement = scaler.fit_transform(displacement)

# Train Test Split
train_input_feature, test_input_feature, train_displacement, test_displacement = train_test_split(input_feature, displacement, test_size=0.2)
train_input_feature, val_input_feature, train_displacement, val_displacement = train_test_split(train_input_feature, train_displacement,test_size=0.2)

# Finding number of input_feature and bcs for the network
num_input_feature = train_input_feature.shape[0]

# Define the inputs to the neural network
train_input_feature = tf.keras.layers.Input(shape=(None,3)) # 3D coordinates of the input_feature

# Concatenate the inputs
inputs = tf.concat([train_input_feature], axis=-1)

# Define the shared MLP and the convolutional layers for feature extraction
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(64, kernel_size=1, activation='relu', input_shape=(None,3)))
model.add(tf.keras.layers.Conv1D(64, kernel_size=1, activation='relu'))
model.add(tf.keras.layers.Conv1D(64, kernel_size=1, activation='relu'))

# Define the max pooling layer
model.add(tf.keras.layers.GlobalMaxPool1D())

# Define the fully connected layers
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation=None))

# Define the displacement field & Apply the model to the inputs
displacement_predicted = model(inputs)

# Split the input_feature tensor into two tensors, one for the point coordinates and one for the boundary conditions
train_points, train_boundary_conditions = tf.split(train_input_feature, [3, 2], axis=-1)

# Compute the gradients with respect to the points tensor
strain = tf.gradients(displacement_predicted, train_points)[0]

# Define the material parameters
mu = tf.keras.layers.Input(tf.float32, shape=()) # Lame's first parameter
lmbda = tf.keras.layers.Input(tf.float32, shape=()) # Lame's second parameter

# Define the right Cauchy-Green deformation tensor
C = tf.matmul(tf.transpose(strain, [0,2,1]), strain)

# Define the first invariant of C
I1 = tf.trace(C)

# Define the Jacobian
J = tf.linalg.det(tf.eye(3) + strain)

# Define the ground substance and volumetric forms of the second Piola-Kirchoff stress tensor
S_gs = (mu/J)*(tf.eye(3) - (I1/3)*tf.linalg.inv(C))
S_vol = (lmbda/J)*(tf.log(J))*tf.eye(3)
S = S_gs + S_vol

# Define the residual forces
force = tf.compat.v1.gradients(S, train_points)[0]

# Define the loss function
loss = tf.reduce_mean(tf.square(force)) + tf.reduce_mean(tf.square(displacement_predicted - train_displacement))

model.compile(loss = loss, optimizer = 'Adam', metrics = ['MeanSquaredError'])

# Define a log directory to store the TensorBoard logs
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Create a TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train_input_feature, train_displacement, epochs=100, batch_size=32, validation_data = (val_input_feature, val_displacement), 
callbacks=[tensorboard_callback])

# Save the model
model.save('my_model.h5')


