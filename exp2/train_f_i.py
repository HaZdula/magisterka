import json
import uuid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import os


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


class SimpleClassifier2(nn.Module):
    def __init__(self, embedding_size=32):
        super(SimpleClassifier2, self).__init__()

        self.fc1 = nn.Linear(embedding_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


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
    id = "test"  # params_dict['id']  # uuid.uuid4()
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    epochs = int(params_dict['epochs'])
    classifier = params_dict['model']
    save_model = params_dict['save_model']
    w_train = params_dict['w_train']
    learning_rate = params_dict['learning_rate']
    momentum = params_dict['momentum']
    checkpoint_path = params_dict['checkpoint_path']

    optimizer = optim.SGD(classifier.parameters(), lr=learning_rate, momentum=momentum)

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
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    if save_model:
        torch.save(classifier.state_dict(), checkpoint_path + f"clsf_{id}.pth")
        # add entry to models_manager in checkpoints
        # with open(checkpoint_path + "models_manager.csv", "a") as f:
        #     f.write(f"{id}, clsf_{id}.pth, {json.dumps(params_dict)}\n")

    classifier.eval()
    return id, classifier


def train_embedding(subset_range=None,
                    epochs_=5,
                    learning_rate_=0.0002,
                    model_class=SimpleClassifier,
                    embedding_size_=32,
                    model_checkpoint_name="f1"):
    if subset_range is None:
        subset_range = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    # convert subset_range to tensor
    subset_range = torch.tensor(list(subset_range))

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    # get subset

    indices = [i for i, target in enumerate(train_dataset.targets) if target in subset_range]
    subset_train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_loader = DataLoader(dataset=subset_train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = model_class(embedding_size_).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_)
    print(f"Training embedding on {subset_range}")
    # Training loop
    for epoch in range(epochs_):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, embedding = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs_}, Loss: {loss.item()}")

    torch.save(model.state_dict(), checkpoint_path + model_checkpoint_name + ".pth")
    return model


checkpoint_path = "./checkpoints/"
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
embedding_size = 32

### f1 i f2 ###

if os.path.exists(checkpoint_path + "f1.pth"):
    model1 = SimpleClassifier(embedding_size).to(device)
    model1.load_state_dict(torch.load(checkpoint_path + "f1.pth"))
    model1.eval()
else:
    model1 = train_embedding(subset_range={0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, model_checkpoint_name="f1")

if os.path.exists(checkpoint_path + "f2.pth"):
    model2 = SimpleClassifier(embedding_size).to(device)
    model2.load_state_dict(torch.load(checkpoint_path + "f2.pth"))
    model2.eval()
else:
    model2 = train_embedding(subset_range={0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, model_checkpoint_name="f2")


def F(data, w_):
    ret = w_ * model1(data)[1] + (1 - w_) * model2(data)[1]
    return ret


def lr_sweep(params_dict_, lr_sweep_range):
    models = []
    accs = []
    for lr in lr_sweep_range:
        params_dict_['learning_rate'] = lr
        print(f"Training classifier on w={params_dict_['w_train']}, lr={params_dict_['learning_rate']}")
        cid, clsf = train_classifier(params_dict_)
        acc = test(w=params_dict_['w_train'], model=clsf, F=F)
        print(f"Accuracy: {acc}")
        models.append(clsf)
        accs.append(acc)
    return models, accs



"""
### BASE TRAINING AND PLOT

print(f"Training classifier on w={w_train}")
cid, classifier = train_classifier(params_dict)

accs = []
weigts = []
for w in np.linspace(0, 1, 11):
    accs.append(test(w=w, model=classifier, F=F))
    weigts.append(w)

plt.plot(weigts, accs)
plt.xlabel("w")
plt.ylabel("accuracy")
# add line at w_train
plt.axvline(x=w_train, color='r', linestyle='--')
plt.show()

"""

### PRETRAINED MODEL

# params
w_pretrain = 0.4
epochs_pretrain = 10
epochs_transfer = 10
w_prime = 0.5
lr_sweep_range = [0.0001, 0.0005, 0.001, 0.005, 0.01]

learning_rate = 0.05
momentum = 0.9

classifier = SimpleClassifier2(embedding_size).to(device)

params_dict = {'w_train': w_pretrain,
               'learning_rate': learning_rate,
               'momentum': momentum,
               'epochs': 10,
               'batch_size': 64,
               'embedding_size': 32,
               'save_model': True,
               'checkpoint_path': "./checkpoints/",
               'device': 'cuda' if torch.cuda.is_available() else 'cpu',
               'model': classifier
               }

if not os.path.exists(checkpoint_path + "pretrained.pth"):
    params_dict['w_train'] = w_pretrain
    params_dict['model'] = classifier
    params_dict['epochs'] = epochs_pretrain
    models, accs = lr_sweep(params_dict, lr_sweep_range)
    # select best model
    best_model = models[np.argmax(accs)]
    best_model.eval()
    torch.save(best_model.state_dict(), checkpoint_path + "pretrained.pth")

    #print(f"Training pretrained classifier on w={params_dict['w_train']}, lr={params_dict['learning_rate']}")
    #cid_pretrained, pretrained_clsf = train_classifier(params_dict)
    #torch.save(pretrained_clsf.state_dict(), checkpoint_path + "pretrained.pth")

else:
    pretrained_clsf = SimpleClassifier2(embedding_size).to(device)
    pretrained_clsf.load_state_dict(torch.load(checkpoint_path + "pretrained.pth"))
    pretrained_clsf.eval()

### LP
if not os.path.exists(checkpoint_path + "lp.pth"):
    lp_clsf = SimpleClassifier2(embedding_size).to(device)
    lp_clsf.load_state_dict(torch.load(checkpoint_path + "pretrained.pth"))
    # freeze all layers except last one
    for param in lp_clsf.parameters():
        print(param)
        param.requires_grad = False
    # unfreeze last layer
    for param in lp_clsf.fc3.parameters():
        param.requires_grad = True

    params_dict['w_train'] = w_prime
    #params_dict['learning_rate'] = 1e-3
    params_dict['model'] = lp_clsf
    params_dict['epochs'] = epochs_transfer
    print(f"Training LP classifier")
    models, accs = lr_sweep(params_dict, lr_sweep_range)
    lp_clsf = models[np.argmax(accs)]
    lp_clsf.eval()
    torch.save(lp_clsf.state_dict(), checkpoint_path + "lp.pth")

else:
    lp_clsf = SimpleClassifier2(embedding_size).to(device)
    lp_clsf.load_state_dict(torch.load(checkpoint_path + "lp.pth"))
    lp_clsf.eval()

### LPFT
if not os.path.exists(checkpoint_path + "lpft.pth"):
    lpft_clsf = SimpleClassifier2(embedding_size).to(device)
    lpft_clsf.load_state_dict(torch.load(checkpoint_path + "pretrained.pth"))

    # freeze all layers except last one
    for param in lpft_clsf.parameters():
        param.requires_grad = False
    # unfreeze last layer
    for param in lp_clsf.fc3.parameters():
        param.requires_grad = True

    params_dict['model'] = lpft_clsf
    params_dict['epochs'] = epochs_transfer / 2
    #params_dict['learning_rate'] = 1e-4
    #print(f"Training LPFT classifier on {params_dict}")
    #_, lpft_clsf = train_classifier(params_dict)  # n epochs
    print(f"Training LP(FT) classifier")
    models, accs = lr_sweep(params_dict, lr_sweep_range)
    lpft_clsf = models[np.argmax(accs)]

    params_dict['model'] = lpft_clsf
    # unfreeze all layers
    for param in lpft_clsf.parameters():
        param.requires_grad = True

    print(f"Training (LP)FT classifier")
    models, accs = lr_sweep(params_dict, lr_sweep_range)
    lpft_clsf = models[np.argmax(accs)]
    torch.save(lpft_clsf.state_dict(), checkpoint_path + "lpft.pth")
else:
    lpft_clsf = SimpleClassifier2(embedding_size).to(device)
    lpft_clsf.load_state_dict(torch.load(checkpoint_path + "lpft.pth"))
    lpft_clsf.eval()

### FT
if not os.path.exists(checkpoint_path + "ft.pth"):
    ft_clsf = SimpleClassifier2(embedding_size).to(device)
    ft_clsf.load_state_dict(torch.load(checkpoint_path + "pretrained.pth"))
    params_dict['model'] = ft_clsf
    params_dict['optimizer'] = optim.SGD(ft_clsf.parameters(), lr=learning_rate, momentum=momentum)
    params_dict['epochs'] = epochs_transfer
    #print(f"Training FT classifier on {params_dict}")
    #_, ft_clsf = train_classifier(params_dict)
    print(f"Training FT classifier")
    models, accs = lr_sweep(params_dict, lr_sweep_range)
    ft_clsf = models[np.argmax(accs)]
    torch.save(ft_clsf.state_dict(), checkpoint_path + "ft.pth")
else:
    ft_clsf = SimpleClassifier2(embedding_size).to(device)
    ft_clsf.load_state_dict(torch.load(checkpoint_path + "ft.pth"))
    ft_clsf.eval()


### TEST

def test_and_plot(model, F, label):
    accs_ = []
    print(f"Testing {label}")
    for w_ in np.linspace(0, 1, 11):
        accs_.append(test(w=w_, model=model, F=F))
    plt.plot(np.linspace(0, 1, 11), accs_, label=label)
    return None


test_and_plot(model=lp_clsf, F=F, label="LP")
test_and_plot(model=lpft_clsf, F=F, label="LPFT")
test_and_plot(model=ft_clsf, F=F, label="FT")
test_and_plot(model=pretrained_clsf, F=F, label="pretrained")

plt.legend()
plt.xlabel("w")
plt.ylabel("accuracy")
plt.axvline(x=w_pretrain, color='r', linestyle='--')
plt.axvline(x=w_prime, color='b', linestyle='--')
plt.show()
