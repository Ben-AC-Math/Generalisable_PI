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
from tensorflow.keras import initializers

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

batch_size=32

# Normalize the data
scaler = MinMaxScaler()
input_feature = scaler.fit_transform(input_feature)
displacement = scaler.fit_transform(displacement)

# Train Test Split
train_input_feature, test_input_feature, train_displacement, test_displacement = train_test_split(input_feature, displacement, test_size=0.2)
train_input_feature, val_input_feature, train_displacement, val_displacement = train_test_split(train_input_feature, train_displacement,test_size=0.2)

# Finding number of input_feature and bcs for the network
num_input_feature = train_input_feature.shape[0]

#Define Initialiser 
initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

# Define the input layer
input_feature = tf.keras.layers.Input(shape=(num_input_feature, 5))

# Define the shared MLP layers for the input transform network
input_transform_mlp1 = tf.keras.layers.Dense(64, kernel_initializer=initializer, activation='relu', name='input_transform_mlp1')(input_feature)
input_transform_mlp2 = tf.keras.layers.Dense(128, kernel_initializer=initializer, activation='relu', name='input_transform_mlp2')(input_transform_mlp1)
input_transform_mlp3 = tf.keras.layers.Dense(128, kernel_initializer=initializer, activation='relu', name='input_transform_mlp3')(input_transform_mlp2)
# Define the max pooling layer
#input_transform_pool = tf.keras.layers.GlobalMaxPooling1D()(input_transform_mlp3)


# # Learn a transformation matrix from the input transform network output
# input_transformation = tf.keras.layers.Dense(5*5, activation='relu', name='input_transformation')(input_transform_mlp2)
# input_transformation = tf.keras.layers.Reshape((6268, 5, 5))(input_transformation)

# # Apply the input transformation matrix to the input point cloud
# transformed_input = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([input_feature, input_transformation])
transformed_input = input_transform_mlp3

# Define shared MLP layers for feature extraction
mlp1 = tf.keras.layers.Dense(64, kernel_initializer=initializer, activation='relu', name='mlp1')(transformed_input)
mlp2 = tf.keras.layers.Dense(64, kernel_initializer=initializer, activation='relu', name='mlp2')(mlp1)

# Define the shared MLP layers for the feature transform network
feature_transform_mlp1 = tf.keras.layers.Dense(64, kernel_initializer=initializer, activation='relu', name='feature_transform_mlp1')(mlp2)
feature_transform_mlp2 = tf.keras.layers.Dense(64, kernel_initializer=initializer, activation='relu', name='feature_transform_mlp2')(feature_transform_mlp1)

# Learn a transformation matrix from the feature transform network output
# feature_transformation = tf.keras.layers.Dense(64*64, activation='relu', name='feature_transformation')(feature_transform_mlp2)
# feature_transformation = tf.keras.layers.Reshape((64, 64))(feature_transformation)

# Apply the feature transformation matrix to the extracted features
# transformed_features = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([mlp2, feature_transformation])
transformed_features = feature_transform_mlp2

# Define additional MLP layers for feature extraction
mlp3 = tf.keras.layers.Dense(64, kernel_initializer=initializer, activation='relu', name='mlp3')(transformed_features)
mlp4 = tf.keras.layers.Dense(128, kernel_initializer=initializer, activation='relu', name='mlp4')(mlp3)
mlp5 = tf.keras.layers.Dense(1024, kernel_initializer=initializer, activation='relu', name='mlp5')(mlp4)

# Define the max pooling layer
#global_feature = tf.keras.layers.GlobalMaxPooling1D()(mlp5)

# Define the fully connected layers
fc1 = tf.keras.layers.Dense(1024, kernel_initializer=initializer, activation='relu', name='fc1')(mlp5)
fc2 = tf.keras.layers.Dense(512, kernel_initializer=initializer, activation='relu', name='fc2')(fc1)
fc3 = tf.keras.layers.Dense(256, kernel_initializer=initializer, activation='relu', name='fc3')(fc2)
fc4 = tf.keras.layers.Dense(128, kernel_initializer=initializer, activation='relu', name='fc4')(fc3)
fc5 = tf.keras.layers.Dense(128, kernel_initializer=initializer, activation='relu', name='fc5')(fc4)

# Define the output layer
output = tf.keras.layers.Dense(3, kernel_initializer=initializer, activation='linear', name='output')(fc5)

model = tf.keras.Model(inputs=input_feature, outputs=output)


def custom_loss(train_input_feature, input_feature, train_displacement, mu, lmbda):
    # Split the input_feature tensor into two tensors, one for the point coordinates and one for the boundary conditions
    input_ph_reshaped = tf.reshape(train_input_feature , [-1, 5])
    train_points = tf.slice(input_ph_reshaped, [0, 0], [-1, 3])

    # Define the displacement field & Apply the model to the inputs
    displacement_predicted = model(train_input_feature)

    # Compute the strain tensor
    strain = tf.gradients(displacement_predicted, train_points)[0]

    print(strain)
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

loss = custom_loss(train_input_feature , input_feature, train_displacement, mu, lmbda) 
model.compile(loss = loss, optimizer = 'Adam', metrics = ['MeanSquaredError'])

# Define a log directory to store the TensorBoard logs
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Create a TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train_input_feature, train_displacement, epochs=100, batch_size=batch_size, validation_data = (val_input_feature, val_displacement), 
callbacks=[tensorboard_callback])

# Save the model
model.save('my_model.h5')


