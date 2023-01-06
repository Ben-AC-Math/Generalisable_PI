
import numpy as np
import pandas as pd

xyz_data = np.array(pd.read_csv('Ux.txt', delimiter = "\t"))[:,1:4]

# Reading displacements and combining into one array
ux = np.array(pd.read_csv('Ux.txt', delimiter = "\t"))[:,4]
uy = np.array(pd.read_csv('Uy.txt', delimiter = "\t"))[:,4]
uz = np.zeros((uy.shape[0],1))
displacement = np.concatenate((ux[:,np.newaxis], uy[:,np.newaxis], uz), axis=1)

sxx = np.array( pd.read_csv('Sxx.txt', delimiter = "\t"))[:,4]
syy = np.array( pd.read_csv('Syy.txt', delimiter = "\t"))[:,4]
sxy = np.array( pd.read_csv('Sxy.txt', delimiter = "\t"))[:,4]
col_data = np.array(pd.read_csv('Collocation Points.txt', delimiter = "\t"))[:,1:3]

bc_data = np.array(pd.read_csv('Ux.txt', delimiter = "\t"))
bc_data = np.concatenate((bc_data, uy[:,np.newaxis]), axis=1)

#filter BC based on if Y coord is 10 
#bc_data_top = bc_data[bc_data[:,2] == 10]
#bc_data_bottom = bc_data[bc_data[:,2] == 0]
#bc_data = np.concatenate((bc_data_top, bc_data_bottom), axis=0)


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

# Concatenate the old array and the new column to create the new array
boundary_conditions = np.concatenate((bc_present, boundary_conditions), axis=1)

print(boundary_conditions.shape, xyz_data.shape, displacement.shape)

#print(bc_present)