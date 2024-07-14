import json
import uuid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Autoencoder(nn.Module):
    def __init__(self, code_len=4):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=3, padding=2),
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=4, stride=3, padding=0),
            nn.ReLU(True),
            nn.Flatten()
            )

        self.fc1 = nn.Linear(16, code_len)
        self.fc2 = nn.Linear(code_len, 16)

        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(16,8,kernel_size=4, stride=3, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(8,4,kernel_size=4, stride=2, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(4,1,kernel_size=5, stride=3, padding=2),
            nn.Sigmoid())
        
    def encode(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        return x

    def decode(self,x):
        x = self.fc2(x)
        x = self.decoder(x.view(-1, 16, 1, 1))
        return x    

    def forward(self,x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    

    
class Autoencoder(nn.Module):
    def __init__(self, code_len=4):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(True),
            nn.Flatten()
            )

        self.fc1 = nn.Linear(128, code_len)
        self.fc2 = nn.Linear(code_len, 128)

        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=2),
            nn.Sigmoid())
        
    def encode(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        return x

    def decode(self,x):
        x = self.fc2(x)
        x = self.decoder(x.view(-1, 128, 1, 1))
        return x    

    def forward(self,x):
        x = self.encode(x)
        x = self.decode(x)
        return x

def load_mnist(batch_size=64, subset_1 = [1, 2, 3, 4, 5, 6, 0], subset_2 = [1, 2, 3, 8, 5, 6, 7]):
    transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.5,), (0.5,))])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # split the dataset
    train_dataset_1 = []
    train_dataset_2 = []
    for inputs, labels in train_loader:
        for i in range(len(labels)):
            if labels[i] in subset_1:
                train_dataset_1.append((inputs[i], labels[i]))
            if labels[i] in subset_2:
                train_dataset_2.append((inputs[i], labels[i]))

    return train_dataset_1, train_dataset_2

def visualize_sample(images, model):
    # visualize original and reconstructed images
    # images = train_dataset_1[:10]
    reconstructed = model(torch.stack([image[0] for image in images]).to(device))

    f, ax = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        print()
        ax[0][i].imshow(images[i][0].squeeze(), cmap='gray')
        ax[1][i].imshow(reconstructed[i].detach().cpu().numpy().squeeze(), cmap='gray')
    plt.show()

    return