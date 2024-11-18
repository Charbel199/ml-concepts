import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import ToTensor
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
# Visualization function
def visualize_predictions(model, loader, device, num_samples=4):
    model.eval()
    data_iter = iter(loader)
    images, masks = next(data_iter)
    images, masks = images.to(device), masks.to(device)
    
    with torch.no_grad():
        predictions = model(images)
        predictions = torch.argmax(predictions, dim=1)  # Convert logits to class indices

    # Plot the results
    fig, axs = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    for i in range(num_samples):
        img = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC for visualization
        mask_gt = masks[i].cpu().numpy().squeeze()
        mask_pred = predictions[i].cpu().numpy()

        axs[i, 0].imshow((img - img.min()) / (img.max() - img.min()))  # Normalize for display
        axs[i, 0].set_title("Input Image")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(mask_gt, cmap="gray")
        axs[i, 1].set_title("Ground Truth")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(mask_pred, cmap="gray")
        axs[i, 2].set_title("Predicted Mask")
        axs[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


# Define the simplified UNet
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(UNet, self).__init__()
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))

        # Decoder
        dec2 = self.decoder2(torch.cat((self.upconv2(bottleneck), enc2), dim=1))
        dec1 = self.decoder1(torch.cat((self.upconv1(dec2), enc1), dim=1))

        # Final output
        return self.final_conv(dec1)


# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 20
NUM_CLASSES = 3   # Background, outline, content


# Dataset transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),         # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Target transformation to ensure proper masks
target_transform = transforms.Compose([
    transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Lambda(lambda x: torch.as_tensor(np.array(x), dtype=torch.long) - 1),  # Subtract 1 from all values
])

# Load the Oxford Pets dataset
dataset = OxfordIIITPet(
    root="dataset/",
    split="trainval",  # Use trainval split for combined training and validation
    target_types="segmentation",
    download=True,
    transform=transform,
    target_transform=target_transform,
)


# Split into train and val datasets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model, loss, and optimizer
model = UNet(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# Training loop
def train_fn(loader, model, optimizer, loss_fn, device):
    model.train()
    loop = tqdm(loader, leave=True)
    for batch_idx, (data, targets) in enumerate(loop):
        data, targets = data.to(device), targets.long().to(device)
        # Forward pass
        outputs = model(data)
        loss = loss_fn(outputs, targets)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        loop.set_postfix(loss=loss.item())


# Evaluation loop
def check_accuracy(loader, model, device):
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            preds = torch.argmax(preds, dim=1)
            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Accuracy: {num_correct / num_pixels * 100:.2f}%")
    print(f"Dice Score: {dice_score / len(loader):.4f}")
    model.train()


# Main training and testing loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train_fn(train_loader, model, optimizer, criterion, DEVICE)
    check_accuracy(val_loader, model, DEVICE)

    # Visualize predictions on validation data
    print("Visualizing predictions...")
    visualize_predictions(model, val_loader, DEVICE)
    
