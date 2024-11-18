import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Create a synthetic dataset with more noise
torch.manual_seed(42)
X = torch.randn(2000, 20)  # Smaller dataset
y = (X[:, 0] * 2 - X[:, 1] + X[:, 2] * 3 - X[:, 3] * 0.5).unsqueeze(1) + 1.5 * torch.randn(2000, 1)  # More noise

# Split into training, validation, and test sets
train_X, val_X, test_X = X[:1200], X[1200:1600], X[1600:]
train_y, val_y, test_y = y[:1200], y[1200:1600], y[1600:]

# Define a more complex model
class OverfitModel(nn.Module):
    def __init__(self):
        super(OverfitModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 300),  # Larger hidden layers
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Training function with early stopping
def train_with_early_stopping(
    model, optimizer, criterion, train_X, train_y, val_X, val_y, test_X, test_y, 
    epochs=100, patience=10
):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(train_X)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X)
            val_loss = criterion(val_outputs, val_y)
            val_losses.append(val_loss.item())
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()  # Save the best model
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load the best model
    model.load_state_dict(best_model)
    
    # Final testing
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_X)
        test_loss = criterion(test_outputs, test_y).item()
    
    return train_losses, val_losses, test_loss

# Initialize models and optimizers
model_no_es = OverfitModel()
model_with_es = OverfitModel()

optimizer_no_es = optim.Adam(model_no_es.parameters(), lr=0.001)
optimizer_with_es = optim.Adam(model_with_es.parameters(), lr=0.001)

criterion = nn.MSELoss()

# Train the models
train_losses_no_es, val_losses_no_es, test_loss_no_es = train_with_early_stopping(
    model_no_es, optimizer_no_es, criterion, train_X, train_y, val_X, val_y, test_X, test_y, 
    epochs=100, patience=float('inf')  # No early stopping
)

train_losses_es, val_losses_es, test_loss_es = train_with_early_stopping(
    model_with_es, optimizer_with_es, criterion, train_X, train_y, val_X, val_y, test_X, test_y, 
    epochs=100, patience=20  # Early stopping with patience 10
)

# Plot the training and validation loss
plt.figure(figsize=(12, 6))

# Training Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses_no_es, label='Train Loss (No Early Stopping)')
plt.plot(train_losses_es, label='Train Loss (With Early Stopping)')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Training Loss')
plt.legend()

# Validation Loss
plt.subplot(1, 2, 2)
plt.plot(val_losses_no_es, label='Validation Loss (No Early Stopping)')
plt.plot(val_losses_es, label='Validation Loss (With Early Stopping)')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Compare final test losses
print(f"Final Test Loss (No Early Stopping): {test_loss_no_es:.4f}")
print(f"Final Test Loss (With Early Stopping): {test_loss_es:.4f}")

'''
No Early Stopping:

The model will continue training even after validation loss stops improving, leading to overfitting.
Validation and test loss will increase after some point.

With Early Stopping:

Training stops early once validation loss stops improving.
Test loss will be lower compared to the model without early stopping.
'''