import random
import torch
from d2l import torch as d2l


class SyntheticRegressionData(d2l.DataModule):  # @save
    """Synthetic data for linear regression."""

    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise

input_dim = 2

@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train + self.num_val))
    for i in range(0, len(indices), self.batch_size):
        batch_indices = torch.tensor(indices[i: i + self.batch_size])
        yield self.X[batch_indices], self.y[batch_indices]


data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
print('features:', data.X[0], '\nlabel:', data.y[0])
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)


import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # One output unit for regression.

    def forward(self, x):
        return self.linear(x)

learning_rate = 0.01

model = LinearRegressionModel(input_dim)  # input_dim is the number of input features.
criterion = nn.MSELoss()  # Mean Squared Error loss for regression.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training parameters
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x)

    # Compute the loss
    loss = criterion(outputs, y)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss at every few epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# After training, you can access the model's parameters
# The weight and bias values are contained in model.linear.weight and model.linear.bias
print("Trained model parameters:")
print("Weight: ", model.linear.weight.item())
print("Bias: ", model.linear.bias.item())
