import random
from d2l import torch as d2l
import torch
import torch.nn as nn


class SyntheticRegressionData(d2l.DataModule):
    """Synthetic data for linear regression."""

    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=64):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise

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


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # One output unit for regression.

    def forward(self, x):
        return self.linear(x)


learning_rate = 0.01
w = [2, -3.4, 7, 7, 8]
input_dim = len(w)
bias = 4.2
data = SyntheticRegressionData(w=torch.tensor(w), b=bias)

model = LinearRegressionModel(input_dim)  # input_dim is the number of input features.
criterion = nn.MSELoss()  # Mean Squared Error loss for regression.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training parameters
num_epochs = 10
# Training loop
for epoch in range(num_epochs):
    total_loss = 0

    # Iterate through the dataset
    for X_batch, y_batch in data.train_dataloader():
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print the average loss for the epoch
    avg_loss = total_loss / len(X_batch)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

model.eval()
# After training, you can access the model's parameters
# The weight and bias values are contained in model.linear.weight and model.linear.bias
print("Trained model parameters:")
print("Weight: ", model.linear.weight)
print("Bias: ", model.linear.bias)


# test on id data
id_data = SyntheticRegressionData(w=torch.tensor(w), b=bias, num_train=0, num_val=1000)
id_loss = 0
for X_batch, y_batch in id_data.val_dataloader():
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    id_loss += loss.item()
id_loss /= len(X_batch)
print(f"ID loss: {id_loss:.4f}")

# testing on ood data
ood_data = SyntheticRegressionData(w=torch.tensor(w), b=bias + 3, num_train=0, num_val=1000)
ood_loss = 0
for X_batch, y_batch in ood_data.val_dataloader():
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    ood_loss += loss.item()
ood_loss /= len(X_batch)
print(f"OOD loss: {ood_loss:.4f}")

# testing on ood data
ood_data = SyntheticRegressionData(w=torch.tensor(w), b=bias + 7, num_train=0, num_val=1000)
ood_loss = 0
for X_batch, y_batch in ood_data.val_dataloader():
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    ood_loss += loss.item()
ood_loss /= len(X_batch)
print(f"OOD loss: {ood_loss:.4f}")






# Fine tune on ID data ??? add bias maybe
criterion = nn.MSELoss()  # Mean Squared Error loss for regression.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# w stays the same w = torch.tensor([2, -3.4])
bias += 3
data = SyntheticRegressionData(w=torch.tensor(w), b=bias)

# Training parameters
num_epochs = 10
# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0

    # Iterate through the dataset
    for X_batch, y_batch in data.train_dataloader():
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print the average loss for the epoch
    avg_loss = total_loss / len(X_batch)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

model.eval()
# After training, you can access the model's parameters
# The weight and bias values are contained in model.linear.weight and model.linear.bias
print("Trained model parameters:")
print("Weight: ", model.linear.weight)
print("Bias: ", model.linear.bias)

# test on id data
id_data = SyntheticRegressionData(w=torch.tensor(w), b=bias, num_train=0, num_val=1000)
id_loss = 0
for X_batch, y_batch in id_data.val_dataloader():
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    id_loss += loss.item()
id_loss /= len(X_batch)
print(f"ID loss: {id_loss:.4f}")

# testing on ood data
ood_data = SyntheticRegressionData(w=torch.tensor(w), b=bias + 7, num_train=0, num_val=1000)
ood_loss = 0
for X_batch, y_batch in ood_data.val_dataloader():
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    ood_loss += loss.item()
ood_loss /= len(X_batch)
print(f"OOD loss: {ood_loss:.4f}")
