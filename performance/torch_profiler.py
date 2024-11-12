import torch
import torch.nn as nn
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T



transform = T.Compose(
    [T.Resize(224),
     T.ToTensor(),
     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

device = torch.device("cuda:0")


# Bigger CNN model
#model = torchvision.models.resnet18().cuda(device)
# Smaller CNN model
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1)
        self.fc1 = nn.Linear(32 * 13 * 13, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)  # Downsample by 2
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)  # Downsample by 2 again
        x = x.view(-1, 32 * 13 * 13)  # Flatten for fully connected layer
        x = self.fc1(x)
        return x
model = SmallCNN().to(device)

criterion = torch.nn.CrossEntropyLoss().cuda(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.train()


def train(data):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet183'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    for step, batch_data in enumerate(train_loader):
        if step >= 1 + 1 + 3:
            break
        train(batch_data)
        prof.step()  # Need to call this at each step to notify profiler of steps' boundary.

# prof = torch.profiler.profile(
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
#         record_shapes=True,
#         with_stack=True)
# prof.start()
# for step, batch_data in enumerate(train_loader):
#     prof.step()
#     if step >= 1 + 1 + 3:
#         break
#     train(batch_data)
# prof.stop()