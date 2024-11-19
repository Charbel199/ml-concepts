import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Create a simple dataset
torch.manual_seed(52)
X = torch.randn(500, 10)
y = (X[:, 0] + X[:, 1] * 2 + X[:, 2] * 3).unsqueeze(1) + 0.1 * torch.randn(500, 1)

# Split into training and test sets
train_X, test_X = X[:400], X[400:]
train_y, test_y = y[:400], y[400:]

# Define a simple model without Dropout
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

# Define a model with Dropout
class DropoutModel(nn.Module):
    def __init__(self):
        super(DropoutModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Dropout(0.3),  # Apply dropout with p=0.5
            nn.Linear(50, 50),
            #nn.BatchNorm1d(50), # Uncomment here for major improvements
            nn.ReLU(),
            nn.Dropout(0.3),  # Apply dropout with p=0.5
            nn.Linear(50, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Training function
def train_model(model, optimizer, criterion, train_X, train_y, test_X, test_y, epochs=100):
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(train_X)
        loss = criterion(outputs, train_y)
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

# Initialize models, optimizers, and loss function
model_no_dropout = SimpleModel()
model_dropout = DropoutModel()

optimizer_no_dropout = optim.Adam(model_no_dropout.parameters(), lr=0.01)
optimizer_dropout = optim.Adam(model_dropout.parameters(), lr=0.01)

criterion = nn.MSELoss()

# Train both models
train_losses_no_dropout, test_losses_no_dropout = train_model(
    model_no_dropout, optimizer_no_dropout, criterion, train_X, train_y, test_X, test_y, epochs=100
)
train_losses_dropout, test_losses_dropout = train_model(
    model_dropout, optimizer_dropout, criterion, train_X, train_y, test_X, test_y, epochs=100
)

# Plot Train Loss and Test Loss with and without Dropout
plt.figure(figsize=(12, 6))

# Train Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses_no_dropout, label='No Dropout')
plt.plot(train_losses_dropout, label='With Dropout')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.legend()

# Test Loss
plt.subplot(1, 2, 2)
plt.plot(test_losses_no_dropout, label='No Dropout')
plt.plot(test_losses_dropout, label='With Dropout')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test Loss')
plt.legend()

plt.tight_layout()
plt.show()



'''
Without dropout, the model tends to overfit, showing a low training loss but a high test loss.
With dropout, the training loss is slightly higher, but the test loss is lower, demonstrating better generalization.
'''