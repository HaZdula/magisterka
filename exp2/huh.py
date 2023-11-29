import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.models as models

# load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

# create simple model for embeddings production
import torch
import torch.nn as nn
import torch.optim as optim

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        # Modify the final fully connected layer to have 10 output units
        resnet.fc = nn.Linear(resnet.fc.in_features, 10)
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)


class ResNet18MNIST(nn.Module):
    def __init__(self):
        super(ResNet18MNIST, self).__init__()
        resnet18 = models.resnet18(pretrained=True)

        # Modify the first convolutional layer to accept 1 input channel
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet18.conv1.weight.sum(dim=1, keepdim=True))

        # Use the rest of the ResNet-18 model (excluding the first layer)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[1:])

        # Add a new fully connected layer with 10 output units
        self.fc = nn.Linear(resnet18.fc.in_features, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet18(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Create an instance of the ResNet18MNIST model
model_18 = ResNet18MNIST()

model_simple = SimpleNN()

model = CustomResNet()


def get_embeddings(dataset, model):
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            embeddings.append(outputs)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


def get_batch_embeddings(data_batch, model):
    with torch.no_grad():
        outputs = model(data_batch)
        embeddings.append(outputs)
    # embeddings = torch.cat(embeddings, dim=0)
    return outputs


model_resnet = models.resnet18()

counter = 0
stop_counter = 100
embeddings = []

train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=False)

for data, label in train_dataloader:
    counter += 1
    if counter == stop_counter:
        break
    #embedding = get_batch_embeddings(data, model_simple)
    embedding = get_batch_embeddings(data, model_)
    embeddings.append(embedding)

# create PCA for embeddings to visualize it
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
converted_embeddings = [embedding[0].numpy() for embedding in embeddings]
pca.fit(converted_embeddings)
embeddings_pca = pca.transform(converted_embeddings)

plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=train_dataset.targets[:len(embeddings_pca)])
plt.show()
