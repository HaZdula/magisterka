import argparse
import os

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights

import utils

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--epochs', default=1, type=int, help='number of epochs tp train for, dividable by 2 for LPFT')
parser.add_argument('--flavor', type=str, help='LP, LPFT, FT')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

args = parser.parse_args()

num_epochs = args.epochs
flavor = args.flavor
lr = args.lr
momentum = args.momentum

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

transform = transforms.Compose(
    [transforms.ToTensor(),
     # normalize if needed
     ])

# load datasets
trainloader10, testloader10 = utils.load_cifar_10_dataset(download=False)
_, testloader101 = utils.load_cifar_10_1_dataset(download=False)

# Pretrained model
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)  # vgg11_bn(pretrained=True)

if flavor == "LPFT":
    num_epochs = int(num_epochs / 2)

if "LP" in flavor:
    # Freeze all layers except the final layer
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    utils.train(model, trainloader10, num_epochs, lr, momentum)

if "FT" in flavor:
    # unfreeze all params
    for param in model.parameters():
        param.requires_grad = True

    utils.train(model, trainloader10, num_epochs, lr, momentum)

print('Finished Training')

cifar10_acc = utils.test(model, testloader10)
cifar101_acc = utils.test(model, testloader101)


results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# save results to file
with open(f"{results_dir}/{flavor}.csv", "a") as f:
    # write header if file is empty
    if os.stat(f"{results_dir}/{flavor}.csv").st_size == 0:
        f.write("cifar10_acc;cifar101_acc;num_epochs;lr;momentum\n")
    f.write(f"{cifar10_acc};{cifar101_acc};{num_epochs};{lr};{momentum}\n")
