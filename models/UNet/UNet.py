import torchvision.transforms.functional as TF
from tqdm import tqdm
import torch.nn as nn
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from .data_loader import get_loaders


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

    def train_fn(self, loader, optimizer, loss_fn, device='cpu'):
        loop = tqdm(loader)

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=device)
            targets = targets.long().to(device=device)

            # forward

            predictions = self(data)
            loss = loss_fn(predictions, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())

    def check_accuracy(self, loader, device="cuda"):
        num_correct = 0
        num_pixels = 0
        dice_score = 0
        self.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.long().to(device)
                predictions = self(x)
                predictions = torch.nn.functional.softmax(predictions, dim=1)
                predictions = torch.argmax(predictions, dim=1)

                num_correct += (predictions == y).sum()
                num_pixels += torch.numel(predictions)
                dice_score += (2 * (predictions * y).sum()) / (
                        (predictions + y).sum() + 1e-8
                )

        print(
            f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
        )
        print(f"Dice score: {dice_score / len(loader)}")
        self.train()

    def save_predictions_as_images(
            self, loader, folder="saved_images", device="cuda"
    ):
        self.eval()
        for idx, (x, y) in enumerate(loader):
            x = x.to(device=device)
            with torch.no_grad():
                predictions = self(x)
                predictions = torch.nn.functional.softmax(predictions, dim=1)
                predictions = torch.argmax(predictions, dim=1)

            ## Save image code here
            ##

        self.train()

    def main(self):
        LEARNING_RATE = 1e-4
        DEVICE = "cpu"
        BATCH_SIZE = 4
        NUM_EPOCHS = 5
        NUM_WORKERS = 0
        IMAGE_HEIGHT = 80  # 1280 originally
        IMAGE_WIDTH = 80  # 1918 originally
        PIN_MEMORY = True
        LOAD_MODEL_FROM_CHECKPOINT = False
        MODEL_CHECKPOINT = "my_checkpoint.pth.tar"
        TRAIN_IMG_DIR = ""
        VAL_IMG_DIR = ""
        PRED_IMG_DIR = ""
        NUM_OF_CLASSES = 11
        SAVE = False
        LOAD = False
        MODEL_PATH = 'unet.pkl'

        model = UNET(in_channels=4, out_channels=NUM_OF_CLASSES).to(DEVICE)
        train_transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

        val_transforms = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_loader, val_loader = get_loaders(
            TRAIN_IMG_DIR,
            VAL_IMG_DIR,
            BATCH_SIZE,
            train_transform,
            val_transforms,
            NUM_WORKERS,
            PIN_MEMORY,
        )

        if not LOAD:

            if LOAD_MODEL_FROM_CHECKPOINT:
                load_checkpoint(torch.load(MODEL_CHECKPOINT), model)

            model.check_accuracy(val_loader, device=DEVICE)

            for epoch in range(NUM_EPOCHS):
                model.train_fn(train_loader, optimizer, loss_fn)

                # save model
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint)

                # check accuracy
                model.check_accuracy(val_loader, device=DEVICE)

                # print some examples to a folder
                model.save_predictions_as_images(
                    val_loader, folder=PRED_IMG_DIR, device=DEVICE
                )
        else:
            model = torch.load(MODEL_PATH)
            model.eval()
            # check accuracy
            model.check_accuracy(val_loader, device=DEVICE)

            # print some examples to a folder
            model.save_predictions_as_images(
                val_loader, folder=PRED_IMG_DIR, device=DEVICE
            )

        if SAVE:
            torch.save(model, MODEL_PATH)
