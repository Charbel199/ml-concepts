import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform['train'])
val_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform['val'])

# Filter only 'cat' (label 3) and 'dog' (label 5) classes
class_indices = {'cat': 3, 'dog': 5}
train_indices = [i for i, label in enumerate(train_dataset.targets) if label in class_indices.values()]
val_indices = [i for i, label in enumerate(val_dataset.targets) if label in class_indices.values()]

train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

# Map labels to 0 ('cat') and 1 ('dog')
for subset in [train_dataset, val_dataset]:
    for i, data in enumerate(subset):
        if subset.dataset.targets[subset.indices[i]] == 3:  # 'cat'
            subset.dataset.targets[subset.indices[i]] = 0
        elif subset.dataset.targets[subset.indices[i]] == 5:  # 'dog'
            subset.dataset.targets[subset.indices[i]] = 1

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load a pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Modify the final layer for binary classification
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # Two classes: cat and dog
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and validation function
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=5):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_losses[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, "
              f"Val Accuracy: {100 * correct / total:.2f}%")

    return train_losses, val_losses

# Train the model
train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, epochs=5)

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
