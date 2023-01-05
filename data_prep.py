
import numpy as np
import pandas as pd

xy_data = np.array(pd.read_csv('Ux.txt', delimiter = "\t"))[:,1:4]
ux = np.array(pd.read_csv('Ux.txt', delimiter = "\t"))[:,4]
uy = np.array(pd.read_csv('Uy.txt', delimiter = "\t"))[:,4]
sxx = np.array( pd.read_csv('Sxx.txt', delimiter = "\t"))[:,4]
syy = np.array( pd.read_csv('Syy.txt', delimiter = "\t"))[:,4]
sxy = np.array( pd.read_csv('Sxy.txt', delimiter = "\t"))[:,4]
col_data = np.array(pd.read_csv('Collocation Points.txt', delimiter = "\t"))[:,1:3]

bc_data = np.array(pd.read_csv('Ux.txt', delimiter = "\t"))
bc_data = np.concatenate((bc_data, uy[:,np.newaxis]), axis=1)

#filter BC based on if Y coord is 10 
bc_data_top = bc_data[bc_data[:,2] == 10]
bc_data_bottom = bc_data[bc_data[:,2] == 0]
bc_data = np.concatenate((bc_data_top, bc_data_bottom), axis=0)

