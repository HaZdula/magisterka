import os
import time

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet50_Weights

from PyTorch_CIFAR10.cifar10_models.vgg import vgg11_bn
from cifar10_1_dataset import CIFAR10_1
import torchvision.models as models
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--epochs', default=1, type=int, help='number of epochs tp train for')

args = parser.parse_args()
num_epochs = args.epochs

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


def test(model, testloader):
    model.eval()  # Set the model to evaluation mode
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
    print(f'Accuracy: {test_epoch_accuracy * 100}% on CIFAR-10')
    return test_epoch_accuracy


# Pretrained model
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)  # vgg11_bn(pretrained=True

transform = transforms.Compose(
    [transforms.ToTensor(),
     ])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Freeze all layers except the final layer
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print('Finished Training')

cifar10_acc = test(model, testloader)

cifar101 = CIFAR10_1("./data", download=True, transform=transform)

testset = CIFAR10_1(root='./data',
                    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

cifar101_acc = test(model, testloader)

results_dir = "results"
if not os.path.exists(results_dir):
    # Create the directory if it doesn't exist
    os.makedirs(results_dir)

# save results to file with timestamp
with open(f"results/LP_{int(time.time())}.txt", "w") as f:
    f.write(f"CIFAR-10:{cifar10_acc}\nCIFAR-10.1:{cifar101_acc}\n")
