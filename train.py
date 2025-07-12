import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from CNN import CNN
from data_setup import *
#from tqdm import tqdm

device = "cuda"
input_size = 784
num_classes = 10
LR = 0.001
num_epochs = 10

model = CNN(in_channels=1, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LR)

for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    for batch_index, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Optimization step
        optimizer.step()
        
        
        