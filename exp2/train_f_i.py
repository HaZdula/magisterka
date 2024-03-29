import json
import uuid

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import os


def test(w, model, F):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(F(inputs, w))
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total} on w={w}")
    return correct / total


def train_classifier(params_dict):
    id = "simple1" #uuid.uuid4()
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    criterion = params_dict['criterion']
    optimizer = params_dict['optimizer']
    # add early stopping
    early_stopping = params_dict['early_stopping']
    epochs = params_dict['epochs']
    classifier = params_dict['model']
    save_model = params_dict['save_model']
    w_train = params_dict['w_train']

    classifier.train()

    # Training loop
    prev_loss = np.inf
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            inp = F(inputs, w_train)
            logits = classifier(inp)
            loss = criterion(logits, labels)
            loss.backward()
            if early_stopping and prev_loss < loss.item():
                break

        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    if save_model:
        torch.save(classifier.state_dict(), checkpoint_path + f"clsf_{id}.pth")
        # add entry to models_manager in checkpoints
        # with open(checkpoint_path + "models_manager.csv", "a") as f:
        #     f.write(f"{id}, clsf_{id}.pth, {json.dumps(params_dict)}\n")

    classifier.eval()
    return id, classifier


# Define the CNN model
class SimpleClassifier(nn.Module):
    def __init__(self, embedding_size=32):
        super(SimpleClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        embedding = torch.relu(self.fc1(x))
        logits = self.fc2(embedding)
        return logits, embedding


model_saved = True
checkpoint_path = "./checkpoints/"

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
embedding_size = 32
learning_rate = 0.0002
epochs = 5

# MNIST jest 28x28

if not model_saved:
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # F1 model, na wszyskich cyfrach
    # Initialize the model, loss function, and optimizer
    model = SimpleClassifier(embedding_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, embedding = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    torch.save(model.state_dict(), checkpoint_path + "model1.pth")

    # Access embeddings for some test data
    test_data = next(iter(train_loader))[0][:5].to(device)
    _, test_embeddings = model(test_data)

    # F2 model na cyfrach 0 i 1

    # select only 0 and 1
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_dataset.data = train_dataset.data[train_dataset.targets < 2]
    train_dataset.targets = train_dataset.targets[train_dataset.targets < 2]

    # Initialize the model, loss function, and optimizer
    model2 = SimpleClassifier(embedding_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model2.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, embedding = model2(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    torch.save(model2.state_dict(), checkpoint_path + "model2.pth")

    # Access embeddings for some test data
    test_data = next(iter(train_loader))[0][:5].to(device)
    _, test_embeddings = model2(test_data)
else:
    model = SimpleClassifier(embedding_size).to(device)
    model.load_state_dict(torch.load(checkpoint_path + "model1.pth"))
    model.eval()
    model2 = SimpleClassifier(embedding_size).to(device)
    model2.load_state_dict(torch.load(checkpoint_path + "model2.pth"))
    model2.eval()


def F(data, w):
    ret = w * model(data)[1] + (1 - w) * model2(data)[1]
    return ret


# chcemy dobrać sobie różne w, takie by symulować różnie przesunięte rozkłady - w=0 to rozkład z modelu 2, w=1 to rozkład z modelu 1
# i spojrzeć jak działa LP FT LPFT

classifier = nn.Sequential(
                   nn.Linear(embedding_size, 28),
                   nn.ReLU(),
                   nn.Linear(28, 28),
                   nn.Linear(28, 10),
                   nn.ReLU(),
               ).to(device)
w_train = 0.5
learning_rate = 0.05
momentum = 0.9
params_dict = {'w_train': w_train,
               'training_subset': {0, 1},
               'learning_rate': learning_rate,
               'momentum': momentum,
               'epochs': 10,
               'batch_size': 64,
               'embedding_size': 32,
               'save_model': True,
               'checkpoint_path': "./checkpoints/",
               'device': 'cuda' if torch.cuda.is_available() else 'cpu',
               'model': classifier,
               'criterion': nn.CrossEntropyLoss(),
               'optimizer': optim.SGD(classifier.parameters(), lr=learning_rate, momentum=momentum),
               'early_stopping': False,
               }

# wizard = pd.read_csv("./checkpoints/models_manager.csv")
# check if param_dict column already contains params_dict
# select column with param_dict, check if any row contains params_dict

# if wizard.loc[wizard.loc[['param_dict']] == json.dumps(params_dict)].shape[0] > 0:
#     print("Model already trained")
#     cid = wizard.loc[wizard['param_dict'] == json.dumps(params_dict)]['id']
#     classifier.load_state_dict(torch.load(checkpoint_path + f"clsf_{cid}.pth"))
# else:

cid, classifier = train_classifier(params_dict)
print(f"Training classifier on w={w_train}")


accs = []
weigts = []
for w in np.linspace(0, 1, 11):
    accs.append(test(w=w, model=classifier, F=F))
    weigts.append(w)

import matplotlib.pyplot as plt
plt.plot(weigts, accs)
plt.xlabel("w")
plt.ylabel("accuracy")
# add line at w_train
plt.axvline(x=w_train, color='r', linestyle='--')
plt.show()

results_dir = "./results/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

with open(results_dir + "results.csv", "a") as f:
    f.write(f"{cid}, {weigts}, {accs}\n")