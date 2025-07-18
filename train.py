"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import engine, model_builder, utils

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torchvision import transforms

device = "cuda" #if torch.cuda.is_available() else "cpu"
input_size = 784
num_classes = 10
LR = 0.001
NUM_EPOCHS = 10

batch_size = 64
train_dataset = datasets.MNIST(root="dataset/", download=True, train=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", download=True, train=False, transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = model_builder.CNN(in_channels=1, num_classes=num_classes).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 
                             lr = LR)

engine.train(model=model,
             train_dataloader=train_loader,
             test_dataloader=test_loader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

utils.save_model(model=model,
                 target_dir="models",
                 model_name="testing_MNIST.pth")
