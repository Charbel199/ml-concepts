import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Create a simple dataset
torch.manual_seed(42)
X = torch.randn(500, 10)
y = (X[:, 0] + X[:, 1] * 2 + X[:, 2] * 3).unsqueeze(1) + 0.1 * torch.randn(500, 1)

# Define a simple model without Batch Normalization
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Define a model with Batch Normalization
class BatchNormModel(nn.Module):
    def __init__(self):
        super(BatchNormModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Training function
def train_model(model, optimizer, criterion, X, y, epochs=100):
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

# Initialize models, optimizers, and loss function
model_no_bn = SimpleModel()
model_bn = BatchNormModel()

optimizer_no_bn = optim.Adam(model_no_bn.parameters(), lr=0.01)
optimizer_bn = optim.Adam(model_bn.parameters(), lr=0.01)

criterion = nn.MSELoss()

# Train both models
losses_no_bn = train_model(model_no_bn, optimizer_no_bn, criterion, X, y, epochs=100)
losses_bn = train_model(model_bn, optimizer_bn, criterion, X, y, epochs=100)

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(losses_no_bn, label='Without Batch Norm')
plt.plot(losses_bn, label='With Batch Norm')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Effect of Batch Normalization on Training Loss')
plt.legend()
plt.show()
