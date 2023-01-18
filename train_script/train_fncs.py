# how to calculate the loss for each epoch => https://discuss.pytorch.org/t/plotting-loss-curve/42632
def train(train_loader, model, optimizer, criterion):
    # Ensure dropout layers are in train mode
    model.train()
    
    running_loss = 0.0
    running_chamfer_loss = 0.0
    running_l1_loss = 0.0
    running_laplacian_loss = 0.0

    # Batches
    # here train_loader returns only x since we use CustomDataset calss and __getitem__ returns only x
    for i_batch, data in enumerate(train_loader):
        # Set device options
        data = data.to(device)

        # Zero gradients
        optimizer.zero_grad()

        y_hat = model(data)
        y = data.y.to(device)
        input_coords = data.pos.to(device)

        multitask_loss, l1_loss, laplacian_loss = criterion(y_hat, y, input_coords, data.edge_index)
        multitask_loss.backward()

        optimizer.step()
        # this can be used if your all batches have equal number of data points including the last batch
        # running_loss += loss.item()
        # if the last batch is smaller than the others
        # data.num_graphs gives the number of graphs per data or batch
        running_loss += multitask_loss.item() * data.num_graphs
        #running_chamfer_loss += chamfer_loss.item() * data.num_graphs
        running_l1_loss += l1_loss.item() * data.num_graphs
        running_laplacian_loss += laplacian_loss.item() * data.num_graphs

    # average training loss => total loss for each epoch (compute by taking sum of each batch loss) devide by total number of batches (i.e. len(train_loader))
    # loss = running_loss / len(train_loader)
    # If you just would like to plot the loss for each epoch, divide the running_loss by the number of batches and append it to loss_values in each epoch.
    # Note, that this might give you a slightly biased loss if the last batch is smaller than the others, so let me know if you need the exact loss.
    # len(train_loader.dataset) gives total number of graphs in the training dataset
    loss = running_loss / len(train_loader.dataset)
    #loss_chamfer = running_chamfer_loss / len(train_loader.dataset)
    loss_l1 = running_l1_loss / len(train_loader.dataset)
    loss_laplacian = running_laplacian_loss / len(train_loader.dataset)
    
    return loss, loss_l1, loss_laplacian
	
	
	
def valid(val_loader, model, criterion):
    model.eval()  # eval mode (no dropout or batchnorm)
    
    running_val_loss = 0.0
    running_l1_loss = 0.0
    running_laplacian_loss = 0.0

    with torch.no_grad():
        # Batches
        # here val_loader returns only x since we use CustomDataset class and __getitem__ returns only x
        for i_batch, data in enumerate(val_loader):
            # Set device options
            data = data.to(device)
            # model(x) returns both encoded and decoded vectors hence we need to define two variables to hold these two values.
            # _ to hold encoded vector and y_hat to hold reconstructed image
            y_hat = model(data)
            y = data.y.to(device)
            input_coords = data.pos.to(device)

            multitask_loss, l1_loss, laplacian_loss = criterion(y_hat, y, input_coords, data.edge_index)
            
            running_val_loss += multitask_loss.item() * data.num_graphs
            running_l1_loss += l1_loss.item() * data.num_graphs
            running_laplacian_loss += laplacian_loss.item() * data.num_graphs

        val_loss = running_val_loss / len(val_loader.dataset)
        loss_l1 = running_l1_loss / len(val_loader.dataset)
        loss_laplacian = running_laplacian_loss / len(val_loader.dataset)

    return val_loss, loss_l1, loss_laplacian
	
	
	
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
        os.makedirs(folder)def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
		
		
def adjust_learning_rate(optimizer, shrink_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
	
	
class CustomMultiTaskLoss(nn.Module):
    def __init__(self):
        super(CustomMultiTaskLoss, self).__init__()
        self.laplacian = DiscreteLaplacianLoss()
        self.l1Loss = nn.L1Loss()
        
    def forward(self, predictedCoords, groundtruthCoords, inputCoords, edge_index):
        laplacianLoss = self.laplacian(predictedCoords, inputCoords, edge_index)
        lossL1 = self.l1Loss(predictedCoords, groundtruthCoords)
        
        return (lossL1 + laplacianLoss), lossL1, laplacianLoss
		
		
lr = 0.0002
patience = 30
start_epoch = 0
epochs = 300
print_freq = 20
save_folder = 'F:/Motion/models_simple/DMM_4_MLPs_with_gantry_info_input_layer_MLP_groupNorm'

model = MotionNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
criterion = CustomMultiTaskLoss()

def modelTrainer():
    best_loss = 100000
    epochs_since_improvement = 0
    train_losses = []
    validation_losses = []
    
    train_l1_losses = []
    train_laplacian_losses = []
    
    validation_l1_losses = []
    validation_laplacian_losses = []

    # Epochs
    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 30:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train_loss, train_loss_l1, train_loss_laplacian = train(train_loader, model, optimizer, criterion)
        train_losses.append(train_loss)
        train_l1_losses.append(train_loss_l1)
        train_laplacian_losses.append(train_loss_laplacian)

        # One epoch's validation
        val_loss, val_loss_l1, val_loss_laplacian = valid(validation_loader, model, criterion)
        validation_losses.append(val_loss)
        validation_l1_losses.append(val_loss_l1)
        validation_laplacian_losses.append(val_loss_laplacian)
        # print('\n * LOSS - {loss:.3f}\n'.format(loss=val_loss_val))
        print('Epoch {} of {}, Train Loss: {:.3f}, Validation Loss: {:.3f}, Train L1 Loss: {:.3f}, Validation L1 Loss: {:.3f}, Train Laplacian Loss: {:.3f}, Validation Laplacian Loss: {:.3f}'.format(epoch, epochs, train_loss, val_loss, train_loss_l1, val_loss_l1, train_loss_laplacian, val_loss_laplacian))

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
    return train_losses, validation_losses, train_l1_losses, validation_l1_losses, train_laplacian_losses, validation_laplacian_losses def modelTrainer():
    best_loss = 100000
    epochs_since_improvement = 0
    train_losses = []
    validation_losses = []
    
    train_l1_losses = []
    train_laplacian_losses = []
    
    validation_l1_losses = []
    validation_laplacian_losses = []

    # Epochs
    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 30:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train_loss, train_loss_l1, train_loss_laplacian = train(train_loader, model, optimizer, criterion)
        train_losses.append(train_loss)
        train_l1_losses.append(train_loss_l1)
        train_laplacian_losses.append(train_loss_laplacian)

        # One epoch's validation
        val_loss, val_loss_l1, val_loss_laplacian = valid(validation_loader, model, criterion)
        validation_losses.append(val_loss)
        validation_l1_losses.append(val_loss_l1)
        validation_laplacian_losses.append(val_loss_laplacian)
        # print('\n * LOSS - {loss:.3f}\n'.format(loss=val_loss_val))
        print('Epoch {} of {}, Train Loss: {:.3f}, Validation Loss: {:.3f}, Train L1 Loss: {:.3f}, Validation L1 Loss: {:.3f}, Train Laplacian Loss: {:.3f}, Validation Laplacian Loss: {:.3f}'.format(epoch, epochs, train_loss, val_loss, train_loss_l1, val_loss_l1, train_loss_laplacian, val_loss_laplacian))

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
    return train_losses, validation_losses, train_l1_losses, validation_l1_losses, train_laplacian_losses, validation_laplacian_losses 
	
	
	
#training_loss_arr, validation_loss_arr, train_chamfer_losses, validation_chamfer_losses, train_l1_losses, validation_l1_losses, train_laplacian_losses, validation_laplacian_losses, train_edge_length_losses, validation_edge_length_losses = modelTrainer()
training_loss_arr, validation_loss_arr, train_l1_losses, validation_l1_losses, train_laplacian_losses, validation_laplacian_losses = modelTrainer()#training_loss_arr, validation_loss_arr, train_chamfer_losses, validation_chamfer_losses, train_l1_losses, validation_l1_losses, train_laplacian_losses, validation_laplacian_losses, train_edge_length_losses, validation_edge_length_losses = modelTrainer()