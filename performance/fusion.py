import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.nn.utils import fuse_conv_bn_eval
import time

# Dataset and Dataloaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./dataset', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./dataset', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define Unfused CNN
class UnfusedCNN(nn.Module):
    def __init__(self):
        super(UnfusedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(36864, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Fused CNN with torch.nn.utils.fusion.fuse_conv_bn_eval
class FusedCNN(UnfusedCNN):
    def __init__(self):
        super(FusedCNN, self).__init__()

    def fuse_model(self):
        # Fuse Conv + BN for both layers
        self.eval()
        self.conv1 = fuse_conv_bn_eval(self.conv1, self.bn1)
        self.conv2 = fuse_conv_bn_eval(self.conv2, self.bn2)
        self.train()
    def forward(self, x):
        x = self.relu1(self.conv1(x))  # Fused conv+bn
        x = self.relu2(self.conv2(x))  # Fused conv+bn
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Fused CNN
class FusedWithReluCNN(UnfusedCNN):
    def __init__(self):
        super(FusedWithReluCNN, self).__init__()

    def fuse_model(self):
        # Fuse Conv + BN + ReLU
        self.eval()
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']], inplace=True)
        self.train()
    def forward(self, x):
        x = self.conv1(x)  # Now includes Conv+BN+ReLU
        x = self.conv2(x)  # Now includes Conv+BN+ReLU
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Training and Evaluation
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    with tqdm(train_loader, desc="Training", unit="batch") as pbar:
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            pbar.set_postfix(loss=total_loss / (total or 1), acc=100. * correct / (total or 1))
    return total_loss / len(train_loader), 100. * correct / total

def test(model, device, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing", unit="batch") as pbar:
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()
                total += target.size(0)

                pbar.set_postfix(loss=total_loss / (total or 1), acc=100. * correct / (total or 1))
    return total_loss / len(test_loader), 100. * correct / total

# Main Function
# Main Function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
unfused_model = UnfusedCNN().to(device)
fused_model = UnfusedCNN().to(device)
#fused_model.fuse_model()  # Perform fusion
fused_with_relu_model = FusedWithReluCNN().to(device)
fused_with_relu_model.fuse_model()  # Perform fusion

# Compile models using torch.compile
unfused_model = torch.compile(unfused_model)
fused_model = torch.compile(fused_model)
fused_with_relu_model = torch.compile(fused_with_relu_model)

# Define loss and optimizers
criterion = nn.CrossEntropyLoss()
optimizer_unfused = optim.SGD(unfused_model.parameters(), lr=0.01, momentum=0.9)
optimizer_fused = optim.SGD(fused_model.parameters(), lr=0.01, momentum=0.9)
optimizer_fused_with_relu = optim.SGD(fused_with_relu_model.parameters(), lr=0.01, momentum=0.9)

# Measure training and inference times
epochs = 2

# Unfused model
start = time.time()
for epoch in range(1, epochs + 1):
    train(unfused_model, device, train_loader, optimizer_unfused, criterion)
unfused_training_time = time.time() - start

start = time.time()
test(unfused_model, device, test_loader, criterion)
unfused_inference_time = time.time() - start

# Fused model
start = time.time()
for epoch in range(1, epochs + 1):
    train(fused_model, device, train_loader, optimizer_fused, criterion)
fused_training_time = time.time() - start

start = time.time()
test(fused_model, device, test_loader, criterion)
fused_inference_time = time.time() - start

# Fused with ReLU model
start = time.time()
for epoch in range(1, epochs + 1):
    train(fused_with_relu_model, device, train_loader, optimizer_fused_with_relu, criterion)
fused_with_relu_training_time = time.time() - start

start = time.time()
test(fused_with_relu_model, device, test_loader, criterion)
fused_with_relu_inference_time = time.time() - start

# Results
print(f"Training Time (Unfused): {unfused_training_time:.4f}s")
print(f"Training Time (Fused): {fused_training_time:.4f}s")
print(f"Training Time (Fused with ReLU): {fused_with_relu_training_time:.4f}s")
print(f"Inference Time (Unfused): {unfused_inference_time:.4f}s")
print(f"Inference Time (Fused): {fused_inference_time:.4f}s")
print(f"Inference Time (Fused with ReLU): {fused_with_relu_inference_time:.4f}s")
