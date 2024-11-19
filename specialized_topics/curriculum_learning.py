import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# Generate synthetic dataset
def generate_data(num_samples, max_value):
    X = torch.randint(0, max_value, (num_samples, 2)).float()
    y = X.sum(dim=1, keepdim=True)
    return X, y

# Define a simple neural network model
class SumModel(nn.Module):
    def __init__(self):
        super(SumModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Training function for curriculum learning
def train_with_curriculum(model, optimizer, criterion, epochs, max_curriculum_value, step):
    train_losses = []
    for epoch in range(epochs):
        current_max_value = min(max_curriculum_value, step * (epoch + 1))
        X, y = generate_data(1000, current_max_value)
        
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        print(f"Curriculum Epoch {epoch+1}/{epochs}, Max Value: {current_max_value}, Loss: {loss.item():.4f}")
    
    return train_losses

# Training function for no curriculum learning
def train_without_curriculum(model, optimizer, criterion, epochs, max_value):
    train_losses = []
    for epoch in range(epochs):
        X, y = generate_data(1000, max_value)  # Full difficulty from the start
        
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        print(f"No Curriculum Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return train_losses

# Initialize models, optimizers, and loss function
torch.manual_seed(42)
model_curriculum = SumModel()
model_no_curriculum = SumModel()

optimizer_curriculum = optim.Adam(model_curriculum.parameters(), lr=0.01)
optimizer_no_curriculum = optim.Adam(model_no_curriculum.parameters(), lr=0.01)

criterion = nn.MSELoss()

# Train both models
epochs = 20
max_curriculum_value = 100  # Maximum difficulty
step = 10  # Increment of difficulty per epoch

print("Training with Curriculum Learning")
curriculum_losses = train_with_curriculum(
    model_curriculum, optimizer_curriculum, criterion, epochs, max_curriculum_value, step
)

print("\nTraining without Curriculum Learning")
no_curriculum_losses = train_without_curriculum(
    model_no_curriculum, optimizer_no_curriculum, criterion, epochs, max_curriculum_value
)

# Plot the training losses
plt.figure(figsize=(10, 5))
plt.plot(curriculum_losses, label="Curriculum Learning")
plt.plot(no_curriculum_losses, label="No Curriculum Learning")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.show()

# Test the models
def test_model(model, max_value):
    model.eval()
    test_X, test_y = generate_data(10, max_value)
    with torch.no_grad():
        predictions = model(test_X)
        print("\nTest Results:")
        for i in range(10):
            print(f"Input: {test_X[i].tolist()}, Target: {test_y[i].item()}, Prediction: {predictions[i].item():.2f}")

print("\nTesting model with Curriculum Learning")
test_model(model_curriculum, max_curriculum_value)

print("\nTesting model without Curriculum Learning")
test_model(model_no_curriculum, max_curriculum_value)
