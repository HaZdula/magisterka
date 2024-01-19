import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

model_saved = True
model_saved_clsf = False
checkpoint_path = "./checkpoints/"

# check if path exists
import os

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)


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

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            logits, embedding = model2(inputs)

            # Calculate the loss
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Update weights
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
# zatem

w_train = 75e-2
learning_rate = 0.0008
momentum = 0.9
epochs = 10

# mlp for classification
if not model_saved_clsf:
    classifier = nn.Sequential(
        nn.Linear(embedding_size, 28),
        nn.ReLU(),
        nn.Linear(28, 10),
        nn.ReLU(),
    ).to(device)

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=learning_rate, momentum=momentum)
        # optim.Adam(classifier.parameters(), lr=learning_rate)
    # add early stopping

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
            optimizer.step()
            if loss.item() < prev_loss:
                prev_loss = loss.item()
            else:
                break
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    torch.save(classifier.state_dict(), checkpoint_path + f"classifier_w_{w_train}_{learning_rate}_{momentum}.pth")
else:
    classifier = nn.Sequential(
        nn.Linear(embedding_size, 28),
        nn.ReLU(),
        nn.Linear(28, 10),
        nn.ReLU(),
    ).to(device)
    classifier.load_state_dict(torch.load(checkpoint_path + f"classifier_w_{w_train}.pth"))
    classifier.eval()


# Test loop
correct = 0
total = 0

classifier.eval()


def test(w, model):
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
    print(f"Accuracy: {100 * correct / total} on w={w}, model_trained_on {w_train}")
    return correct / total


test(w=0.1, model=classifier)
test(w=0.3, model=classifier)
test(w=0.5, model=classifier)
test(w=0.7, model=classifier)
test(w=0.9, model=classifier)
