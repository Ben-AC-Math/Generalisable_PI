# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:13:57 2023

@author: mathe
"""

import tensorflow as tf

# Define the inputs to the neural network
points = tf.placeholder(tf.float32, shape=(None, num_points, 3))  # 3D coordinates of the points
boundary_conditions = tf.placeholder(tf.float32, shape=(None, num_points, num_boundary_conditions))  # Boundary conditions at each point

# Concatenate the inputs
inputs = tf.concat([points, boundary_conditions], axis=-1)

# Define the convolutional and pooling layers
num_filters = 64
kernel_size = 1
stride = 1
padding = "VALID"
pooling_size = num_points

layer = inputs
for i in range(num_hidden_layers):
    layer = tf.layers.conv1d(layer, num_filters, kernel_size, stride, padding)
    layer = tf.layers.max_pooling1d(layer, pool_size=pooling_size, strides=pooling_size)
global_feature = layer

# Define the fully connected layers
num_neurons_per_layer = 32
activation_function = tf.nn.relu

layer = global_feature
for i in range(num_hidden_layers):
    layer = tf.layers.dense(layer, num_neurons_per_layer, activation=activation_function)
output = tf.layers.dense(layer, 3)  # <-- The output is the displacement field

# Define the outputs of the neural network
displacement = output  # <-- The displacement field is the output of the neural network

# Define the loss function
displacement_loss = tf.losses.mean_squared_error(displacement, output)

# Define the physics-informed loss term
neo_hookean_loss_weight = 1e-3  # Weight for the neo-Hookean loss term
stress = displacement * neo_hookean_loss_weight  # Stress at each point
strain = (stress - displacement) / stress  # Strain at each point
F = tf.exp(strain)  # Deformation gradient tensor
J = tf.linalg.det(F)  # Determinant of the deformation gradient tensor
I1 = tf.linalg.trace(F)  # First invariant of the deformation gradient tensor
mu = 1.0  # Lame's constant
neo_hookean_loss = mu / 2.0 * (I1 - 3) - mu * tf.log(J)  # Neo-Hookean loss
loss = displacement_loss + neo_hookean_loss_weight * neo_hookean_loss

# Combine the loss terms
loss = displacement_loss + neo_hookean_loss_weight * ((J - 1)**2 + (I1 - 3)**2)

# Define the optimizer
learning_rate = 1e-3
optimizer = tf.train.AdamOptimizer(learning_rate)

# Define the training operation
training_op = optimizer.minimize(loss)

# Define the number of training steps and the batch size
num_steps = 10000
batch_size = 32

# Create a session to train the neural network
with tf.Session() as sess:
    # Initialize the variables
    sess.run(tf.global_variables_initializer())
    
    # Iterate over the training steps
    for step in range(num_steps):
        # Select a batch of training data
        points_batch, displacement_batch, boundary_conditions_batch = ...
        
        # Run the training operation
        sess.run(training_op, feed_dict={points: points_batch,
                                         displacement: displacement_batch,
                                         boundary_conditions: boundary_conditions_batch})
        
        # Print the loss at each step
        if step % 1000 == 0:
            loss_value = sess.run(loss, feed_dict={points: points_batch,
                                                   displacement: displacement_batch,
                                                   boundary_conditions: boundary_conditions_batch})
            print("Step:", step, "Loss:", loss_value)
            
    # Save the trained model
    saver = tf.train.Saver()
    saver.save(sess, "./point_net_model")
