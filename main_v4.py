import os
import numpy as np
import itertools
import math, random
random.seed = 42
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim

class Tnet(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, input):
        # input.shape == (bs,n,3)
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))

        #initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
        if xb.is_cuda:
            init=init.cuda()
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init
        return matrix

class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=5)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(5,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        out_t2 = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(out_t2)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        output = output.view(-1, 1024, 1).repeat(1, 1, input.shape[2])
        print('output dim:', output.shape)
        print('out_t2 dim:', out_t2.shape)
        concat_out = torch.concat([out_t2, output], dim=1)

        print('concat_out dim:', concat_out.shape)

        return concat_out

class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Conv1d(1088, 512, 1)
        self.fc2 = nn.Conv1d(512, 256, 1)
        self.fc3 = nn.Conv1d(256, 128, 1)
        self.fc4 = nn.Conv1d(128, 128, 1)
        self.fc5 = nn.Conv1d(128, 3, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
    
        #self.dropout = nn.Dropout(p=0.3)

    def forward(self, input):
        print('input shape (PointNet):', input.shape)
        xb = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.fc2(xb)))
        xb = F.relu(self.bn3(self.fc3(xb)))
        xb = F.relu(self.bn4(self.fc4(xb)))

        output = self.fc5(xb)

        return output

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

train_data = np.concatenate((train_input_feature, train_displacement), axis=1)
test_data = np.concatenate((test_input_feature, test_displacement), axis=1)
val_data = np.concatenate((val_input_feature, val_displacement ), axis=1)

# Finding number of input_feature and bcs for the network
num_input_feature = train_input_feature.shape[0]

# how to calculate the loss for each epoch => https://discuss.pytorch.org/t/plotting-loss-curve/42632
def train(train_loader, model, optimizer, criterion):
    # Ensure dropout layers are in train mode
    model.train()
    
    running_loss = 0.0
    running_mse_loss = 0.0
    running_pinn_loss = 0.0

    # Batches
    # here train_loader returns only x since we use CustomDataset calss and __getitem__ returns only x
    for i_batch, data in enumerate(train_loader):
        # Set device options
        input = data[0][:,:5].reshape(1, 5, 6268).float()
        #print('data size:', data.size(0))

        print('input:', input)

        y = data[0][:,-3:].reshape(1, 3, 6268).float()
        print('y:', y)

        print('Coords:', input[:,:3])

        # Zero gradients
        optimizer.zero_grad()

        y_hat = model(input)
      
        #input_coords = data.pos.to(device)

        multitask_loss, mse_loss, pinn_loss = criterion(y_hat, y, input[:,:3])
        multitask_loss.backward()

        optimizer.step()
        # this can be used if your all batches have equal number of data points including the last batch
        # running_loss += loss.item()
        # if the last batch is smaller than the others
        # data.num_graphs gives the number of graphs per data or batch
        running_loss += multitask_loss.item() * data.size(0)
        #running_chamfer_loss += chamfer_loss.item() * data.num_graphs
        running_mse_loss += mse_loss.item() * data.size(0)
        running_pinn_loss += pinn_loss.item() * data.size(0)

    # average training loss => total loss for each epoch (compute by taking sum of each batch loss) devide by total number of batches (i.e. len(train_loader))
    # loss = running_loss / len(train_loader)
    # If you just would like to plot the loss for each epoch, divide the running_loss by the number of batches and append it to loss_values in each epoch.
    # Note, that this might give you a slightly biased loss if the last batch is smaller than the others, so let me know if you need the exact loss.
    # len(train_loader.dataset) gives total number of graphs in the training dataset
    loss = running_loss / len(train_loader.dataset)
    #loss_chamfer = running_chamfer_loss / len(train_loader.dataset)
    loss_mse = running_mse_loss / len(train_loader.dataset)
    loss_pinn = running_pinn_loss / len(train_loader.dataset)
    
    return loss, loss_mse, loss_pinn
	
	
	
def valid(val_loader, model, criterion):
    model.eval()  # eval mode (no dropout or batchnorm)
    
    running_val_loss = 0.0
    running_mse_loss = 0.0
    running_pinn_loss = 0.0

    with torch.no_grad():
        # Batches
        # here val_loader returns only x since we use CustomDataset class and __getitem__ returns only x
        for i_batch, data in enumerate(val_loader):
            # Set device options
            input = data[0][:,:5].float().reshape(1, 5, 1568)
            y = data[0][:,-3:].float().reshape(1, 3, 1568)

            # model(x) returns both encoded and decoded vectors hence we need to define two variables to hold these two values.
            # _ to hold encoded vector and y_hat to hold reconstructed image
            y_hat = model(input)

            multitask_loss, mse_loss, pinn_loss = criterion(y_hat, y, input[:,:3])
            
            running_val_loss += multitask_loss.item() * data.size(0)
            running_mse_loss += mse_loss.item() * data.size(0)
            running_pinn_loss += pinn_loss.item() * data.size(0)

        val_loss = running_val_loss / len(val_loader.dataset)
        loss_mse = running_mse_loss / len(val_loader.dataset)
        loss_pinn = running_pinn_loss / len(val_loader.dataset)

    return val_loss, loss_mse, loss_pinn
	
	
	
def save_checkpoint(epoch, model, optimizer, val_loss, is_best):
    ensure_folder(save_folder)
    state = {'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict()}
    filename = '{0}/checkpoint_{1}_{2:.3f}.tar'.format(save_folder, epoch, val_loss)
    torch.save(model.state_dict(), filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, '{}/BEST_checkpoint.tar'.format(save_folder))
		
		
def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
		
		
def adjust_learning_rate(optimizer, shrink_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
	
	

class PinnLoss(nn.Module):
    def __init__(self):
        super(PinnLoss, self).__init__()
        
    def forward(self, predictedDisp, inputPoints):
      
      strain = torch.autograd.grad(
          predictedDisp, inputPoints,
          grad_outputs=torch.ones_like(predictedDisp),
          retain_graph=True,
          create_graph=True
          )[0]

      f = 5 * strain
        
      return f

class CustomMultiTaskLoss(nn.Module):
    def __init__(self):
        super(CustomMultiTaskLoss, self).__init__()
        self.pinnLoss = PinnLoss()
        self.mseLoss = nn.MSELoss()
        
    def forward(self, predictedDisp, groundtruthDisp, inputPoints):
        pinnLoss = self.pinnLoss(predictedDisp, inputPoints)
        lossMSE = self.mseLoss(predictedDisp, groundtruthDisp)
        
        return (lossMSE + pinnLoss), lossMSE, pinnLoss

lr = 0.0002
patience = 30
start_epoch = 0
epochs = 300
print_freq = 20
save_folder = '/content/drive/MyDrive/PIPN/saved_models'

model = PointNet()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
criterion = CustomMultiTaskLoss()

def modelTrainer():
    best_loss = 100000
    epochs_since_improvement = 0
    train_losses = []
    validation_losses = []
    
    train_mse_losses = []
    train_pinn_losses = []
    
    validation_mse_losses = []
    validation_pinn_losses = []

    # Epochs
    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 30:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train_loss, train_loss_mse, train_loss_pinn = train(train_loader, model, optimizer, criterion)
        train_losses.append(train_loss)
        train_mse_losses.append(train_loss_mse)
        train_pinn_losses.append(train_loss_pinn)

        # One epoch's validation
        val_loss, val_loss_mse, val_loss_pinn = valid(validation_loader, model, criterion)
        validation_losses.append(val_loss)
        validation_mse_losses.append(val_loss_mse)
        validation_pinn_losses.append(val_loss_pinn)
        # print('\n * LOSS - {loss:.3f}\n'.format(loss=val_loss_val))
        print('Epoch {} of {}, Train Loss: {:.3f}, Validation Loss: {:.3f}, Train MSE Loss: {:.3f}, Validation MSE Loss: {:.3f}, Train PINN Loss: {:.3f}, Validation PINN Loss: {:.3f}'.format(epoch, epochs, train_loss, val_loss, train_loss_mse, val_loss_mse, train_loss_pinn, val_loss_pinn))

        # Check if there was an improvement
        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, val_loss, is_best)
    # return train_losses, validation_losses, train_chamfer_losses, validation_chamfer_losses, train_l1_losses, validation_l1_losses, train_laplacian_losses, validation_laplacian_losses, train_edge_length_losses, validation_edge_length_losses 
    return train_losses, validation_losses, train_mse_losses, validation_mse_losses, train_pinn_losses, validation_pinn_losses 

training_loss_arr, validation_loss_arr, train_mse_losses, validation_mse_losses, train_pinn_losses, validation_pinn_losses = modelTrainer()