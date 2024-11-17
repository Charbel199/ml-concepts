import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

class CNN(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 5, kernel_size=(7, 7), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(5, 8, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(12 * 12 * 8, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.network(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 3e-4
momentum = 0.9
batch_size = 32
epochs = 5

# Initialize models
model = CNN(1).to(device)

# Set optimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Loading MNIST dataset
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  # Mean and stdv of MNIST dataset
)

train_dataset = datasets.MNIST(root="dataset/", transform=transforms, train=True, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="dataset/", transform=transforms, train=False, download=True)
test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Prepare models for training
model.train()

for epoch in range(epochs):
    loop = tqdm(train_loader, leave=True)
    for batch_idx, data in enumerate(loop):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update tqdm progress bar
        loop.set_description(f"Epoch [{epoch}/{epochs}]")
        loop.set_postfix(
            loss=loss.item()
        )

data_iter = iter(test_loader)
images, labels = next(data_iter)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
# print images
plt.imshow(images[0][0])
plt.show()
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
outputs = model(images.to(device))
predicted = torch.argmax(outputs, 1).tolist()
print('Predicted: ', ' '.join('%5s' % classes[round(predicted[j])] for j in range(batch_size)))
