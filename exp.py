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
#parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--download', default=False, type=bool, help='download datasets')
parser.add_argument('--moco', type=str, help='path to moco checkpoint')

args = parser.parse_args()

num_epochs = args.epochs
flavor = args.flavor
#lr = args.lr
momentum = args.momentum
download = args.download
moco_path = args.moco

SWEEP_LRS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

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

cifar_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225)),
     ])

stl_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225)),
     ])

# load datasets
trainloader10, testloader10 = utils.load_cifar_10_dataset(download=download, transform=cifar_transform)
_, testloader101 = utils.load_cifar_10_1_dataset(download=download, transform=cifar_transform)
_, testloaderSTL = utils.load_STL_10_dataset(download=download, transform=stl_transform)

# Pretrained model
model = models.resnet50()  # weights=ResNet50_Weights.IMAGENET1K_V2)  # vgg11_bn(pretrained=True)

checkpoint_dict = torch.load(moco_path)['state_dict']

# rename moco pre-trained keys
for k in list(checkpoint_dict.keys()):
    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
        # remove prefix
        checkpoint_dict[k[len("module.encoder_q."):]] = checkpoint_dict[k]
    # delete renamed or unused k
    del checkpoint_dict[k]

model.load_state_dict(checkpoint_dict, strict=False)

loss_dir = "losses"
loss_save_path = f"{loss_dir}/{flavor}.csv"

# create dir if not exists
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)

for lr in SWEEP_LRS:
    print(f"Sweeping lr:{lr}")
    if flavor == "LPFT":
        num_epochs = int(num_epochs / 2)

    if "LP" in flavor:
        # Freeze all layers except the final layer
        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

        utils.train(model, trainloader10, num_epochs, lr, momentum, loss_save_path)

    if "FT" in flavor:
        # unfreeze all params
        for param in model.parameters():
            param.requires_grad = True

        utils.train(model, trainloader10, num_epochs, lr, momentum, loss_save_path)

    # restore num_epochs for saving results
    num_epochs = args.epochs
    print('Finished Training')

    cifar10_acc = utils.test(model, testloader10)
    cifar101_acc = utils.test(model, testloader101)
    stl_acc = utils.test(model, testloaderSTL)

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # save results to file
    with open(f"{results_dir}/{flavor}.csv", "a") as f:
        # write header if file is empty
        if os.stat(f"{results_dir}/{flavor}.csv").st_size == 0:
            f.write("cifar10_acc;cifar101_acc;stl_acc;num_epochs;lr;momentum\n")
        f.write(f"{cifar10_acc};{cifar101_acc};{stl_acc};{num_epochs};{lr};{momentum}\n")

    # save model
    torch.save(model.state_dict(), f"{results_dir}/{flavor}_{lr}.pt")
