import os

import torch
import torchvision.models as models
import torchvision.transforms as transforms

import utils


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

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

    download = False
    # load datasets
    # trainloader10, testloader10 = utils.load_cifar_10_dataset(download=download, transform=cifar_transform)
    # _, testloader101 = utils.load_cifar_10_1_dataset(download=download, transform=cifar_transform)
    #_, testloaderSTL = utils.load_STL_10_dataset(download=download, transform=stl_transform)
    _, testloaderSTL = utils.load_STL_10_relabeled_dataset(download=download, transform=stl_transform)
    checkpoint_dir = "results/checkpoints"

    for f in os.listdir(checkpoint_dir):
        model = models.resnet50()
        checkpoint_dict = torch.load(f)['state_dict']
        model.load_state_dict(checkpoint_dict, strict=False)
        model.eval()

        stl_acc = utils.test(model, testloaderSTL)
        print(f"{f}: {stl_acc}")


if __name__ == '__main__':
    main()
