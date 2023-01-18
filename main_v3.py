import torch
import torch.nn as nn
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

#tf.compat.v1.disable_eager_execution()

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
index = (bc_data[:, 2] == 5)

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

class PointNet(nn.Module):
    def __init__(self, num_input_feature, initializer):
        super(PointNet, self).__init__()
        
        self.num_input_feature = num_input_feature
        self.initializer = initializer
        
        self.input_transform_mlp1 = nn.Linear(5, 64, bias=True, activation=nn.ReLU)
        self.input_transform_mlp2 = nn.Linear(64, 128, bias=True, activation=nn.ReLU)
        self.input_transform_mlp3 = nn.Linear(128, 1024, bias=True, activation=nn.ReLU)
        self.input_transform_pool = nn.AdaptiveMaxPool1d(1)
        self.input_transform_mlp4 = nn.Linear(1024, 512, bias=True, activation=nn.ReLU)
        self.input_transform_mlp5 = nn.Linear(512, 256, bias=True, activation=nn.ReLU)

        self.input_transformation = nn.Linear(128, 5*5, bias=True, activation=nn.ReLU)
        self.transformed_input = torch.matmul
        
        self.mlp1 = nn.Linear(5, 64, bias=True, activation=nn.ReLU)
        self.mlp2 = nn.Linear(64, 64, bias=True, activation=nn.ReLU)
        
        self.feature_transform_mlp1 = nn.Linear(64, 64, bias=True, activation=nn.ReLU)
        self.feature_transform_mlp2 = nn.Linear(64, 128, bias=True, activation=nn.ReLU)
        self.feature_transform_mlp3 = nn.Linear(128, 1024, bias=True, activation=nn.ReLU)
        self.feature_transform_pool = nn.AdaptiveMaxPool1d(1)
        self.feature_transform_mlp4 = nn.Linear(1024, 512, bias=True, activation=nn.ReLU)
        self.feature_transform_mlp5 = nn.Linear(512, 256, bias=True, activation=nn.ReLU)

        self.feature_transformation = nn.Linear(128, 64*64, bias=True, activation=nn.ReLU)
        self.transformed_features = torch.matmul
        
        self.mlp3 = nn.Linear(64, 64, bias=True, activation=nn.ReLU)
        self.mlp4 = nn.Linear(64, 128, bias=True, activation=nn.ReLU)
        self.mlp5 = nn.Linear(128, 1024, bias=True, activation=nn.ReLU)
        
        self.global_feature = nn.AdaptiveMaxPool1d(1)
        
        #add output of T2 to output of global feature 

        self.output1 = self.mlp3(input)
        self.output2 = self.global_feature(input)

        self.combined_output = torch.cat((self.output1, self.output2), dim=1)
        self.fc1 = nn.Linear(1088, 512, bias=True, activation=nn.ReLU)
        self.fc2 = nn.Linear(512, 256, bias=True, activation=nn.ReLU)
        self.fc3 = nn.Linear(256, 128, bias=True, activation=nn.ReLU)
        self.fc4 = nn.Linear(128, 128, bias=True, activation=nn.ReLU)
        
        self.output = nn.Linear(128, 3, bias=True)

# Define the output layer
output = tf.keras.layers.Dense(3, kernel_initializer=initializer, activation='linear', name='output')(fc4)

model = tf.keras.Model(inputs=input_feature, outputs=output)

model.compile(loss = 'mse', optimizer = 'Adam')

batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((train_input_feature, train_displacement)).batch(batch_size)

# define the custom training loop
epochs = 10
for epoch in range(epochs):
    #for step, (train_input_feature, train_displacement) in enumerate(train_dataset):
    for train_input_feature, train_displacement in zip(train_input_feature, train_displacement):

        with tf.GradientTape(persistent=True) as tape:
            train_displacement_pred = model(train_input_feature, training=True)
            
            train_points_reshaped = tf.reshape(train_input_feature , [-1, 5])
            train_points = tf.slice(train_points_reshaped, [0, 0], [-1, 3])

            tape.watch(train_points)

            # Compute the strain tensor
            strain = tape.gradient(train_displacement_pred, train_points)
            
            # print strain tensor
            print(strain)

            loss_value = tf.reduce_mean(tf.square(train_displacement - train_displacement_pred)) + tf.reduce_mean(tf.square(strain))
        
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


# Save the model
model.save('my_model.h5')




