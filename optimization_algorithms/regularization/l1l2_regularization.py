import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Create a larger synthetic dataset
torch.manual_seed(42)
X = torch.randn(5000, 10)
y = (X[:, 0] * 2 - X[:, 1] + X[:, 2] * 3 - X[:, 3] * 0.5).unsqueeze(1) + 0.5 * torch.randn(5000, 1)

# Split into training and test sets
train_X, test_X = X[:4000], X[4000:]
train_y, test_y = y[:4000], y[4000:]

# Define a more complex model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Training function with L1/L2 regularization
def train_model(model, optimizer, criterion, train_X, train_y, test_X, test_y, epochs=100, l1_lambda=0.0, l2_lambda=0.0):
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(train_X)
        loss = criterion(outputs, train_y)
        
        # Add L1 and L2 regularization to the loss
        l1_norm = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        l2_norm = sum(torch.sum(param ** 2) for param in model.parameters())
        loss += l1_lambda * l1_norm + l2_lambda * l2_norm
        
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Testing
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_X)
            test_loss = criterion(test_outputs, test_y)
            test_losses.append(test_loss.item())
    
    return train_losses, test_losses

# Initialize models and optimizers
# Initialize models and optimizers
model_no_reg = SimpleModel()
model_l1 = SimpleModel()
model_l2 = SimpleModel()
model_elastic_net = SimpleModel()  # Add Elastic Net model

optimizer_no_reg = optim.Adam(model_no_reg.parameters(), lr=0.01)
optimizer_l1 = optim.Adam(model_l1.parameters(), lr=0.01)
optimizer_l2 = optim.Adam(model_l2.parameters(), lr=0.01)
optimizer_elastic_net = optim.Adam(model_elastic_net.parameters(), lr=0.01)  # Optimizer for Elastic Net

criterion = nn.MSELoss()

# Train the models
train_losses_no_reg, test_losses_no_reg = train_model(
    model_no_reg, optimizer_no_reg, criterion, train_X, train_y, test_X, test_y, epochs=100
)
train_losses_l1, test_losses_l1 = train_model(
    model_l1, optimizer_l1, criterion, train_X, train_y, test_X, test_y, epochs=100, l1_lambda=1e-2
)
train_losses_l2, test_losses_l2 = train_model(
    model_l2, optimizer_l2, criterion, train_X, train_y, test_X, test_y, epochs=100, l2_lambda=1e-2
)
train_losses_elastic_net, test_losses_elastic_net = train_model(
    model_elastic_net, optimizer_elastic_net, criterion, train_X, train_y, test_X, test_y, epochs=100, l1_lambda=1e-2, l2_lambda=1e-2
)  # Elastic Net Regularization

# Plot the training and test loss
plt.figure(figsize=(12, 6))

# Training Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses_no_reg, label='No Regularization')
plt.plot(train_losses_l1, label='L1 Regularization')
plt.plot(train_losses_l2, label='L2 Regularization')
plt.plot(train_losses_elastic_net, label='Elastic Net Regularization')  # Add Elastic Net
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Training Loss with Regularization')
plt.legend()

# Test Loss
plt.subplot(1, 2, 2)
plt.plot(test_losses_no_reg, label='No Regularization')
plt.plot(test_losses_l1, label='L1 Regularization')
plt.plot(test_losses_l2, label='L2 Regularization')
plt.plot(test_losses_elastic_net, label='Elastic Net Regularization')  # Add Elastic Net
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.title('Test Loss with Regularization')
plt.legend()

plt.tight_layout()
plt.show()



'''
No Regularization: Lower training loss but higher test loss due to overfitting.
L1 Regularization: Higher training loss, better generalization (sparse weights).
L2 Regularization: More stable convergence, penalizes large weights, better generalization.
Elastic Net Regularization: Balances sparsity and smoothness, achieving intermediate or better test loss depending on the dataset.
'''