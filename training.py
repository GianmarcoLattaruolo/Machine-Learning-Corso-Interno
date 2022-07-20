import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm_notebook
from torch.utils.data import Dataset, DataLoader

# Test the network on test dataset to check accuracy on mnist classification
def my_accuracy(net, device, dataloader):
    """
    Compute the classification accuracy of the model and confusion matrix
    ___________
    Parameters:
    net = network
    device = training device(cuda/cpu)
    dataloader = dataloader
  
    ________
    Returns:
    mismatched = list of all mismatched examples
    confusion_list = list whose elements are list of true labels and probabilities of the model
    accuracy = classification accuracy
    """
    # Set evaluation mode
    net.eval()

    total = 0
    correct = 0
    mismatched = []
    predictions = []
    trues = []
    
    with torch.no_grad():
        for  x_batch, label_batch in dataloader:
                
            # Move to device
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(device)

            # Forward pass
            y_hat = net(x_batch)
            y_hat = y_hat.squeeze()

            # Apply softmax 
            softmax = nn.Softmax(dim=0)
            out_soft = softmax(y_hat)

            # Take the prediction
            predicted = out_soft.detach().cpu().argmax().item()
            
            # True value
            true = label_batch.detach().cpu().item()

            if predicted==true:
                correct += 1
            else:
                mismatched.append((x_batch.detach().cpu().numpy(), predicted, true))
                
            # Take probabilities
            prob = out_soft.detach().cpu().numpy()
            
            # Append to lists
            predictions.append(predicted)
            trues.append(true)
                                  
            total += 1

    return mismatched, [trues, predictions], 100.0*correct/total

### Train epoch function
def train_epoch(net, device, dataloader, loss_function, optimizer):
    """
    Train an epoch of data
    -----------
    Parameters:
    net = network
    device = training device (cuda/cpu)
    dataloader = dataloader of data
    loss_function = loss function
    optimzer = optimizer used
    --------
    Returns:
    mean(train_epoch_loss) = average epoch loss
    """
    # Set the train mode
    net.train()
    # List to save batch losses
    train_epoch_loss = []
    # Iterate the dataloader
    for x_batch, label_batch in dataloader:

        # Move to device
        x_batch = x_batch.to(device)
        label_batch = label_batch.to(device)
    
        # Forward pass
        y_hat = net(x_batch)

        # Compute loss
        loss = loss_function(y_hat, label_batch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute batch losses
        train_batch_loss = loss.detach().cpu().numpy()
        train_epoch_loss.append(train_batch_loss)
        
    return np.mean(train_epoch_loss)
    

### Test epoch function
def val_epoch(net,  device, dataloader, loss_function):
    """
    Validate an epoch of data
    -----------
    Parameters:
    net = network
    device = training device (cuda/cpu)
    dataloader = dataloader of data
    loss_function = loss function
    optimzer = optimizer used
    --------
    Returns:
    mean(val_epoch_loss) = average validation loss
    """
    # Set evaluation mode
    net.eval()
    # List to save evaluation losses
    val_epoch_loss = []
    with torch.no_grad():
        for x_batch, label_batch in dataloader:
                
            # Move to device
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(device)

            # Forward pass
            y_hat = net(x_batch)

            # Compute loss
            loss = loss_function(y_hat, label_batch)

            # Compute batch_loss
            val_batch_loss = loss.detach().cpu().numpy()
            val_epoch_loss.append(val_batch_loss)

    return np.mean(val_epoch_loss)

### Training epochs
def train_epochs(net, device, train_dataloader, val_dataloader, loss_function, optimizer, max_num_epochs, early_stopping = False):
    """
    Train an epoch
    ___________
    Parameters:
    max_num_epochs: maximum number of epochs (sweeps tthrough the datasets) to train the model
    early_stopping: if true stop the training if the last validation loss is greater 
    than the average of last 100 epochs
    """
    
    # Progress bar
    pbar = tqdm(range(max_num_epochs))

    # Inizialize empty lists to save losses
    train_loss_log = []
    val_loss_log = []

    for epoch_num in pbar:

        # Train epoch
        mean_train_loss = train_epoch(net, device, train_dataloader, loss_function, optimizer)

        # Validate epoch
        mean_val_loss = val_epoch(net, device, val_dataloader, loss_function)

        # Append losses and accuracy
        train_loss_log.append(mean_train_loss)
        val_loss_log.append(mean_val_loss)

        # Set pbar description
        pbar.set_description("Train loss: %s" %round(mean_train_loss,2)+", "+"Val loss %s" %round(mean_val_loss,2))
        
        # Early stopping
        if early_stopping:
            if epoch_num>10 and np.mean(val_loss_log[-10:]) < val_loss_log[-1]:
                print("Training stopped at epoch "+str(epoch_num)+" to avoid overfitting.")
                break
    
    return train_loss_log, val_loss_log

### Trainin epochs with accuracy (classification tasks)
def train_epochs_acc(net, device, train_dataloader, val_dataloader, test_dataloader, loss_function, optimizer, max_num_epochs, early_stopping = True):
    
    # Pogress bar
    pbar = tqdm_notebook(range(max_num_epochs))

    # Inizialize empty lists to save losses
    train_loss_log = []
    val_loss_log = []
    accuracy = []

    for epoch_num in pbar:
        # Compute accuracy before training
        mismatched, confusion, acc = my_accuracy(net, device, test_dataloader)

        # Tran epoch
        mean_train_loss = train_epoch(net, device, train_dataloader, loss_function, optimizer)

        # Validate epoch
        mean_val_loss = val_epoch(net,  device, val_dataloader, loss_function)

        # Append losses and accuracy
        train_loss_log.append(mean_train_loss)
        val_loss_log.append(mean_val_loss)
        accuracy.append(acc)

        # Set pbar description
        pbar.set_description("Train loss: %s" %round(mean_train_loss,2)+", "+"Val loss %s" %round(mean_val_loss,2)
                             +", "+"Test accuracy %s" %round(acc,2)+"%")
        
        # Early stopping
        if early_stopping:
            if np.mean(val_loss_log[-10:]) < val_loss_log[-1]:
                print("Training stopped at epoch "+str(epoch_num)+" to avoid overfitting.")
                break
    
    return train_loss_log, val_loss_log, accuracy

