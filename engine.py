"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.
    Puts the PyTorch model into training mode and then 
    runs through all the required training steps (forward pass, 
    loss calculation, optimizer step).

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be trained on.
        loss_fn (torch.nn.Module): A PyTorch loss function to minimize.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
        device (torch.device): A target device to compute on (e.g. "cpu" or "cuda")

    Returns:
        Tuple[float, float]: A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy).
    """

    model.train() # Put the model in train mode
    
    train_loss, train_acc = 0,0  # Setup train loss and train accuracy values
    
    # Loop through data loader batches
    for batch, (sample, target) in enumerate(dataloader):
       
        sample, target = sample.to(device), target.to(device)  # Send data to target device

        pred = model(sample) # Forward pass
        
        loss = loss_fn(pred, target) # Calculate loss
        train_loss += loss.item() # Accumulate loss

        optimizer.zero_grad()

        loss.backward() # Backpropagation
        
        optimizer.step()
        
        # Calculate and accumulate accuracy metric across all batches
        pred_class = torch.argmax(torch.softmax(pred, dim=1), dim=1) # Returns the predicted class
        train_acc += (pred_class == target).sum().item() / len(pred) # Calculates acuracy of correctly predicted classes
        
    # Get the average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc
          
          
          
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.
    Turns the PyTorch model to "eval" mode and performs
    a forward pass on a testing dataset.

    Args:
        model (torch.nn.Module): The target PyTorch model to be tested.
        dataloader (torch.utils.data.DataLoader): The testing DataLoader instance for the model to be tested on.
        loss_fn (torch.nn.Module): A PyTorch loss function to calculate loss on the test data.
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        Tuple[float, float]: A touple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_acc).
    """
    model.eval() # Put the model in eval mode
    
    test_loss, test_acc = 0,0  # Setup test loss and test accuracy values
    
    # Turn on interence context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (sample, target) in enumerate(dataloader):
            sample, target = sample.to(device), target.to(device) # Send data to target device
            
            test_pred = model(sample) # Forward pass
            
            loss = loss_fn(test_pred, target) # Calculate loss
            test_loss += loss.item() # Accumulate loss
            
            test_pred_labels = test_pred.argmax(dim=1) # Return the predicted label
            test_acc += ((test_pred_labels == target).sum().item()) / len(test_pred_labels) # Calculate and accumulate accuracy
        
        # Get average loss and accuracy per batch
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model.
    Passes a PyTorch model through train_step() and test_step()
    functions for a number of epochs, training and validating the model
    in the same epoch loop.
    
    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model (torch.nn.Module): A PyTorch model to be trained and tested.
        train_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be trained on.
        test_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be etsted on.
        loss_fn (torch.nn.Module): A PyTorch loss function to calculate loss on both datasets.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer function to help minimize the loss function.
        epochs (int): An integer indicating how many epochs to train for.
        device (torch.device): The target device to compute on.

    Returns:
        Dict[str, List]: A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {"train_loss": [...],
                    "train_acc": [...],
                    "test_loss": [...],
                    "test_acc": [...]} 
    """
    
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        
        # Print what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
        )
        
        # Update the dict
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
    return results
            
    