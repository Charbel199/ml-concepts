import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim



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


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

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
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(train_loader)} \
                      Loss: {loss:.4f}"
            )

data_iter = iter(test_loader)
images, labels = data_iter.next()

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
# print images
plt.imshow(images[0][0])
plt.show()
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
outputs = model(images)
predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))
