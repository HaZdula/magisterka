import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

# load resent-50
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
model.eval()


def get_embeddings(dataset):
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            embeddings.append(outputs)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


train_embeddings = get_embeddings(train_dataset)
test_embeddings = get_embeddings(test_dataset)
