# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:13:57 2023

@author: mathe
"""

# Import necessary libraries
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Import the PointNet++ layers
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D

tf.compat.v1.disable_eager_execution()

#Defining material parameters
mu = 5
lmbda = 50

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

train_input_feature = np.expand_dims(train_input_feature, axis=0)
test_input_feature = np.expand_dims(test_input_feature, axis=0)
val_input_feature = np.expand_dims(val_input_feature, axis=0)

# Define the input layer
input_feature = tf.keras.layers.Input(shape=(num_input_feature, 5))

# Define the encoder layers
encoder1 = tf.keras.layers.Conv1D(64, 1, activation='relu', name='encoder1')(input_feature)
encoder2 = tf.keras.layers.Conv1D(64, 1, activation='relu', name='encoder2')(encoder1)
encoder3 = tf.keras.layers.Conv1D(64, 1, activation='relu', name='encoder3')(encoder2)
encoder4 = tf.keras.layers.Conv1D(128, 1, activation='relu', name='encoder4')(encoder3)
encoder5 = tf.keras.layers.Conv1D(1024, 1, activation='relu', name='encoder5')(encoder4)

# Define the global feature layer
global_feature = tf.keras.layers.GlobalMaxPooling1D()(encoder5)

# Define the output layer
output = tf.keras.layers.Dense(3, activation='linear', name='output')(global_feature)

# Define placeholders for the input and output of the model
input_ph = tf.keras.layers.Input(shape=(train_input_feature.shape[1], train_input_feature.shape[2]))
displacement_ph = tf.keras.layers.Input(shape=(train_displacement.shape[1]))

model = tf.keras.Model(inputs=input_ph, outputs=displacement_ph)

def custom_loss(train_input_feature, input_ph, train_displacement, mu, lmbda):
    # Split the input_feature tensor into two tensors, one for the point coordinates and one for the boundary conditions
    input_ph_reshaped = tf.reshape(train_input_feature , [-1, 5])
    train_points = tf.slice(input_ph_reshaped, [0, 0], [-1, 3])

    # Define the displacement field & Apply the model to the inputs
    displacement_predicted = model(input_ph)

    # Compute the strain tensor
    strain = tf.gradients(displacement_predicted, train_points)[0]

    # Define the right Cauchy-Green deformation tensor
    C = tf.matmul(tf.transpose(strain, [0,2,1]), strain)

    # Define the first invariant of C
    I1 = tf.trace(C)

    # Define the Jacobian
    J = tf.linalg.det(tf.eye(3) + strain)

    # Define the ground substance and volumetric forms of the second Piola-Kirchoff stress tensor
    S_gs = (mu/J)*(tf.eye(3) - (I1/3)*tf.linalg.inv(C))
    S_vol = (lmbda/J)*(tf.math.log(J))*tf.eye(3)
    S = S_gs + S_vol

    # Compute the force
    force = tf.gradients(S, train_points)[0]

    # Define the loss function
    loss = tf.reduce_mean(tf.square(force)) + tf.reduce_mean(tf.square(displacement_predicted - train_displacement))

    return loss

loss = custom_loss(train_input_feature , input_ph, train_displacement, mu, lmbda) 
model.compile(loss = loss, optimizer = 'Adam', metrics = ['MeanSquaredError'])

# Define a log directory to store the TensorBoard logs
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Create a TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(input_ph, displacement_ph, epochs=100, batch_size=32, validation_data = (val_input_feature, val_displacement), 
callbacks=[tensorboard_callback])


# Save the model
model.save('my_model.h5')


