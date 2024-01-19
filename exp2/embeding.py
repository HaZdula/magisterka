import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        # List to store intermediate representations
        embeddings = []

        # Forward pass up to the final average pooling layer
        for layer in self.features:
            x = layer(x)
            embeddings.append(x)

        return embeddings


# Instantiate the model
model_with_embeddings = SimpleNN()

# Set the model to evaluation mode
model_with_embeddings.eval()

# Generate a random input tensor with shape [batch_size, channels, height, width]
#input_tensor = torch.randn(1, 1, 28, 28)

# load mnist dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

# Instantiate the model
model_with_embeddings = SimpleNN()
model_with_embeddings.eval()

num_samples = 1000
subset_indices = list(range(num_samples))
subset = torch.utils.data.Subset(train_dataset, subset_indices)

# pass mnist dataset to model
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(subset_indices))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# List to store the embeddings
intermediate_embeddings = []

# Forward pass up to the final average pooling layer
for images, labels in train_loader:
    # Pass the input through the model
    embeddings = model_with_embeddings(images)
    # Store only the final layer embeddings
    intermediate_embeddings.append(embeddings[1])


#plot pca
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# change to numpy array
concatenated_embeddings = torch.cat(intermediate_embeddings, dim=0)

# change to numpy array
concatenated_embeddings = concatenated_embeddings.detach().numpy()
concatenated_embeddings = concatenated_embeddings.reshape(concatenated_embeddings.shape[0], -1)
# Get the labels
labels = train_dataset.targets.numpy()

# Define the PCA algorithm
pca = PCA(n_components=2)

# Apply PCA to the embeddings
pca_embeddings = pca.fit_transform(concatenated_embeddings)

# Define the figure
fig = plt.figure(figsize=(10, 10))

# Define the axis
ax = fig.add_subplot(1, 1, 1)

# Define the colors to use
colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black", "gray"]

# Loop over the embeddings
for i in range(pca_embeddings.shape[0]):
    # Get the label for the digit corresponding to the current embedding
    label = labels[i]

    # Get the color corresponding to the label
    color = colors[label]

    # Get the x and y coordinates corresponding to the embedding
    x, y = pca_embeddings[i, :]

    # Plot the digit
    ax.scatter(x, y, c=color)

# Show the plot
plt.show()