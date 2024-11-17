import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Discriminator(nn.Module):
    def __init__(self, image_channels, features_d):
        # Features_d is just used for simpler scaling of the convolution neural network
        super(Discriminator, self).__init__()
        # Based on  DCGAN paper
        self.network = nn.Sequential(
            nn.Conv2d(image_channels, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    @staticmethod
    def _block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),  # Bias false for batch norm
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.network(x)


class Generator(nn.Module):
    def __init__(self, noise_dimension, image_channels, features_g):
        super(Generator, self).__init__()
        # Based on  DCGAN paper
        self.network = nn.Sequential(
            self._block(noise_dimension, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Image pixels between -1 and 1
        )

    @staticmethod
    def _block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            # Bias false for batch norm
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)


def initialize_model_weights(model):
    # Initializing weights based on DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# Hyperparameters. Note: GAN very sensitive to hyperparameters (specially simple ones such as this one)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 3e-4
noise_dimension = 100
image_size = 64
image_channels = 1
batch_size = 128
epochs = 5
features_g = 64
features_d = 64

# Initialize models
discriminator = Discriminator(image_channels, features_d).to(device)
initialize_model_weights(discriminator)
generator = Generator(noise_dimension, image_channels, features_g).to(device)
initialize_model_weights(generator)

# Set optimizers
opt_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
opt_generator = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
criterion = nn.BCELoss()  # Loss function

# Used for visualization
fixed_noise = torch.randn(32, noise_dimension, 1, 1).to(device)

# Loading MNIST dataset
transforms = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for x in range(image_channels)], [0.5 for x in range(image_channels)]
        )
    ]
)
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# For tensorboard
writer_fake = SummaryWriter(f"runs/DCGAN_MNIST/fake")  # For fake images
writer_real = SummaryWriter(f"runs/DCGAN_MNIST/real")  # For real images
step = 0

# Prepare models for training
generator.train()
discriminator.train()

for epoch in range(epochs):
    loop = tqdm(loader, leave=True)
    for batch_idx, (real_image, _) in enumerate(loop):
        batch_size = real_image.shape[0]

        ## Fetching real image and generating FAKE image
        # Real image represents the actual MNIST image
        real_image = real_image.to(device)
        # Noise represents a random matrix of noise used by the generator as input
        noise = torch.randn(batch_size, noise_dimension, 1, 1).to(device)
        fake_image = generator(noise)

        ## Train discriminator, maximize (log(D(real)) + log(1-D(G(z))))
        # log(D(real)) part
        discriminator_real = discriminator(real_image).view(-1)
        loss_discriminator_real = criterion(discriminator_real, torch.ones_like(
            discriminator_real))  # Discriminator should associate real image with  1
        # log(1-D(G(z)) part
        discriminator_fake = discriminator(fake_image).view(-1)
        loss_discriminator_fake = criterion(discriminator_fake, torch.zeros_like(
            discriminator_fake))  # Discriminator should associate fake image with  0
        # Combined loss
        loss_discriminator = (loss_discriminator_real + loss_discriminator_fake) / 2

        discriminator.zero_grad()
        # loss.backward() sets the grad attribute of all tensors with requires_grad=True in the computational graph
        loss_discriminator.backward(
            retain_graph=True)  # retrain_graph so that what was used in this pass is not cleared from cache, ex: fake_image
        opt_discriminator.step()

        ## Train generator, minimize log(1-D(G(z))) OR maximize log(D(G(z)))
        # log(D(G(z)) part
        discriminator_fake = discriminator(fake_image).view(-1)
        loss_generator = criterion(discriminator_fake, torch.ones_like(
            discriminator_fake))  # Generator wants Discriminator to associate fake image with  1

        generator.zero_grad()
        loss_generator.backward(
            retain_graph=True)  # retrain_graph so that what was used in this pass is not cleared from cache, ex: fake_image
        opt_generator.step()

        # Update tqdm progress bar
        loop.set_description(f"Epoch [{epoch}/{epochs}]")
        loop.set_postfix(
            loss_discriminator=loss_discriminator.item(),
            loss_generator=loss_generator.item(),
        )

        # Tensorboard code
        if batch_idx == 0:
            with torch.no_grad():
                fake = generator(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real_image[:32], normalize=True)
                writer_fake.add_image("Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Real Images", img_grid_real, global_step=step)

                step += 1
