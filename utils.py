import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from cifar10_1_dataset import CIFAR10_1


def load_cifar_10_dataset(root='./data',
                          download=True,
                          transform=transforms.Compose([transforms.ToTensor()]),
                          batch_size=64):
    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=download, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=download, num_workers=2)
    return trainloader, testloader


def load_cifar_10_1_dataset(root='./data',
                            download=True,
                            transform=transforms.Compose([transforms.ToTensor()]),
                            batch_size=64):
    trainset = CIFAR10_1(root=root,
                         download=download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2)

    testset = CIFAR10_1(root=root,
                        download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2)

    return trainloader, testloader


def load_STL_10_dataset(root='./data',
                            download=True,
                            transform=transforms.Compose([transforms.ToTensor()]),
                            batch_size=64):
    trainset = torchvision.datasets.STL10(root=root, split='train',
                            download=download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2)
    testset = torchvision.datasets.STL10(root=root, split='test',
                            download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2)
    return trainloader, testloader


def test(model, testloader):
    model.eval()
    correct_test_predictions = 0
    total_test_samples = 0

    with torch.no_grad():
        for test_data in testloader:
            test_inputs, test_labels = test_data
            test_outputs = model(test_inputs)

            _, test_predicted = test_outputs.max(1)
            correct_test_predictions += (test_predicted == test_labels).sum().item()
            total_test_samples += test_labels.size(0)

    # Calculate accuracy for the testing epoch
    test_epoch_accuracy = correct_test_predictions / total_test_samples
    print(f'Accuracy: {test_epoch_accuracy * 100}% on {testloader.dataset}')
    return test_epoch_accuracy


def train(model, trainloader, num_epochs, lr=0.001, momentum=0.9, savefile=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    model.train()

    for epoch in tqdm(range(num_epochs)):
        for i, data in tqdm(enumerate(trainloader, 0)):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            l=loss.item()
            with open(savefile, 'a') as f:
                # add header if file is empty
                if os.stat(savefile).st_size == 0:
                    f.write("epoch;loss;lr;momentum\n")
                f.write(f"{epoch};{l};{lr};{momentum}\n")
