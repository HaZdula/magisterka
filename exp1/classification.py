import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# Function to generate synthetic data for two classes with different distributions
def generate_data(num_samples, mean, std):
    return [torch.normal(mean, std) for i in range(num_samples)]



# Class for the synthetic dataset
class SyntheticDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Parameters for the two distributions
num_samples_class1 = 500
num_samples_class2 = 500

mean_class1 = torch.tensor([2.0, 2.0])
std_class1 = torch.tensor([1.0, 1.0])

mean_class2 = torch.tensor([8.0, 8.0])
std_class2 = torch.tensor([2.0, 2.0])

# Generate synthetic data for two classes
data_class1 = generate_data(num_samples_class1, mean_class1, std_class1)
data_class2 = generate_data(num_samples_class2, mean_class2, std_class2)

# Labels for the two classes
labels_class1 = torch.zeros(num_samples_class1, dtype=torch.long)
labels_class2 = torch.ones(num_samples_class2, dtype=torch.long)

# Create datasets for the two classes
dataset_class1 = SyntheticDataset(data_class1, labels_class1)
dataset_class2 = SyntheticDataset(data_class2, labels_class2)

# Combine the datasets to create a dataset with distribution shift
combined_dataset = torch.utils.data.ConcatDataset([dataset_class1, dataset_class2])

# Shuffle the combined dataset
dataloader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

# Visualize the synthetic data
plt.scatter(data_class1[:, 0], data_class1[:, 1], label='Class 1')
plt.scatter(data_class2[:, 0], data_class2[:, 1], label='Class 2')
plt.title('Synthetic Data with Distribution Shift')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
