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


# load mnist dataset and split into two datasets
# one with only 1,2,3,4,5,6,9,0 and the other one 7,2,8,4,5,6,9,0
# keep the labels so that 1 changes into 7 and 3 into 8

def load_mnist(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.5,), (0.5,))])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # split the dataset
    train_dataset_1 = []
    train_dataset_2 = []
    for inputs, labels in train_loader:
        for i in range(len(labels)):
            if labels[i] in [1, 2, 3, 4, 5, 6, 0]:
                train_dataset_1.append((inputs[i], labels[i]))
            if labels[i] in [1, 2, 3, 8, 5, 6, 7]:
                train_dataset_2.append((inputs[i], labels[i]))

    return train_dataset_1, train_dataset_2

def visualize_sample():
    # visualize original and reconstructed images
    images = train_dataset_1[:10]
    reconstructed = model(torch.stack([image[0] for image in images]).to(device))

    f, ax = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        print()
        ax[0][i].imshow(images[i][0].squeeze(), cmap='gray')
        ax[1][i].imshow(reconstructed[i].detach().cpu().numpy().squeeze(), cmap='gray')
    plt.show()

    return


if __name__ == '__main__':
    # Set device (GPU if available, else CPU)
    device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset
    train_dataset_1, train_dataset_2 = load_mnist()

    ae_train_loader_1 = DataLoader(dataset=train_dataset_1[:2000], batch_size=32, shuffle=True)
    ae_test_loader_1 = DataLoader(dataset=train_dataset_1[2000:4000], batch_size=32, shuffle=True)

    # Initialize the model, loss function, and optimizer
    embedding_size = 12 
    model = Autoencoder(embedding_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0017)
    epochs = 150

    # Training loop
    for epoch in range(epochs):
        epoch_losses = []

        model.train()
        for inputs, labels in ae_train_loader_1:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach().cpu().numpy())
        
        val_epoch_losses = []
        model.eval()
        with torch.no_grad():
            for inputs, labels in ae_test_loader_1:
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_epoch_losses.append(loss.detach().cpu().numpy())

        print(f"Epoch {epoch+1}/{epochs} train loss: {np.mean(epoch_losses)} validation loss: {np.mean(val_epoch_losses)}")

    # Training classfier on the second dataset
    #train_loader_2 = DataLoader(dataset=train_dataset_1[:2000], batch_size=32, shuffle=True)
    classifier = nn.Sequential(
                   nn.Linear(embedding_size, 32),
                   nn.ReLU(),
                   nn.Linear(32, 16),
                   nn.ReLU(),
                   nn.Linear(16, 8),
                   nn.Sigmoid()
               ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    epochs = 50

    # rename labels appropriately
    # 1 -> 7 4-8
    # 3 -> 8 0-7
    for i in range(len(train_dataset_2)):
        if train_dataset_2[i][1] == 8:
            train_dataset_2[i] = (train_dataset_2[i][0], torch.tensor(4))
        if train_dataset_2[i][1] == 7:
            train_dataset_2[i] = (train_dataset_2[i][0], torch.tensor(0))

    cl_train_loader_1 = DataLoader(dataset=train_dataset_1[4000:6000], batch_size=32, shuffle=True)
    cl_test_loader_1 = DataLoader(dataset=train_dataset_1[6000:8000], batch_size=32, shuffle=True)

    cl_train_loader_2 = DataLoader(dataset=train_dataset_2[:2000], batch_size=32, shuffle=True)
    cl_test_loader_2 = DataLoader(dataset=train_dataset_2[2000:4000], batch_size=32, shuffle=True)

    # Training loop
    print("### Training classifier")
    for epoch in range(epochs):

        model.train()
        epoch_losses = []
        for inputs, labels in cl_train_loader_1:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(model.encode(inputs))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach().cpu().numpy())

        model.eval()
        correct = 0
        count = 0
        with torch.no_grad():
            for inputs, labels in cl_train_loader_1:
                outputs = classifier(model.encode(inputs))
                correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
                count += len(labels)
            train_accuracy = correct / count

            for inputs, labels in cl_test_loader_1:
                outputs = classifier(model.encode(inputs))
                correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
                count += len(labels)
            test_accuracy = correct / count
                

        print(f"Epoch {epoch+1}/{epochs} loss: {np.mean(epoch_losses)} train accuracy: {train_accuracy} test accuracy: {test_accuracy}")


    # Fine-tuning loop
    print("### Fine-tuning ###")
    for epoch in range(epochs):
        
        model.train()
        epoch_losses = []
        for inputs, labels in cl_train_loader_2:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(model.encode(inputs))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach().cpu().numpy())
        
        model.eval()
        correct = 0
        count = 0
        with torch.no_grad():
            for inputs, labels in cl_train_loader_2:
                outputs = classifier(model.encode(inputs))
                correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
                count += len(labels)
            train_accuracy = correct / count

            for inputs, labels in cl_test_loader_2:
                outputs = classifier(model.encode(inputs))
                correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
                count += len(labels)
            test_accuracy = correct / count

        print(f"Epoch {epoch+1}/{epochs} loss: {np.mean(epoch_losses)} train accuracy: {train_accuracy} test accuracy: {test_accuracy}")



    images = train_dataset_2[:12]
    #print([image[1] for image in images])
    code = (model.encode(torch.stack([image[0] for image in images]).to(device)))
    reconstructed = model.decode(code)
    print(classifier(code).argmax(1))
    f, ax = plt.subplots(2, 12, figsize=(20, 4))
    for i in range(12):
        print()
        ax[0][i].imshow(images[i][0].squeeze(), cmap='gray')
        ax[1][i].imshow(reconstructed[i].detach().cpu().numpy().squeeze(), cmap='gray')
    plt.show()

