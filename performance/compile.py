import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel RGB
    transforms.Resize((224, 224)),  # Resize for ResNet18
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./dataset', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./dataset', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

# Define ResNet18 model
def get_model():
    model = models.resnet18(pretrained=False)  # No pretraining for MNIST
    model.fc = nn.Linear(model.fc.in_features, 10)  # 10 classes for MNIST
    return model.to(device)

# Define training loop
def train_one_epoch(model, loader, criterion, optimizer, desc="Training"):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(loader)

# Define inference loop
def evaluate(model, loader, desc="Evaluating"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Benchmark function with warm-up
def benchmark(model, train_loader, test_loader, epochs=1, use_compile=False, warmup=True):
    if use_compile:
        model = torch.compile(model, mode='reduce-overhead')

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Warm-up run (optional)
    if warmup:
        print("Warm-up Run...")
        train_one_epoch(model, train_loader, criterion, optimizer, desc="Warm-up")

    # Training benchmark
    print("Benchmarking Training...")
    start_time = time.time()
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, desc=f"Epoch {epoch+1}/{epochs}")
    train_time = time.time() - start_time

    # Inference benchmark
    print("Benchmarking Inference...")
    start_time = time.time()
    accuracy = evaluate(model, test_loader, desc="Inference")
    inference_time = time.time() - start_time

    return train_time, inference_time, accuracy

# Run benchmarks
print("Running Benchmark for Uncompiled Model...")
model_uncompiled = get_model()
train_time_uncompiled, inference_time_uncompiled, accuracy_uncompiled = benchmark(model_uncompiled, train_loader, test_loader, epochs=10)

print("\nResults:")
print("Uncompiled Model:")
print(f"Training Time: {train_time_uncompiled:.2f}s, Inference Time: {inference_time_uncompiled:.2f}s, Accuracy: {accuracy_uncompiled:.2%}")

print("\nRunning Benchmark for Compiled Model...")
model_compiled = get_model()
torch._dynamo.reset()
train_time_compiled, inference_time_compiled, accuracy_compiled = benchmark(model_compiled, train_loader, test_loader, epochs=10, use_compile=True)

# Print results
print("\nResults:")
print("\nCompiled Model:")
print(f"Training Time: {train_time_compiled:.2f}s, Inference Time: {inference_time_compiled:.2f}s, Accuracy: {accuracy_compiled:.2%}")
