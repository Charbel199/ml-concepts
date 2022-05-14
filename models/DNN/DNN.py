import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim


class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(image_dimension, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)




# Hyperparameters. Note: GAN very sensitive to hyperparameters (specially simple ones such as this one)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 3e-4

batch_size = 32
epochs = 5

# Initialize models
discriminator = Discriminator(image_dimension).to(device)
generator = Generator(noise_dimension, image_dimension).to(device)
# Set optimizers
opt_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate)
opt_generator = optim.Adam(generator.parameters(), lr=learning_rate)
criterion = nn.BCELoss()  # Loss function

# Used for visualization
fixed_noise = torch.randn((batch_size, noise_dimension)).to(device)

# Loading MNIST dataset
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  # Mean and stdv of MNIST dataset
)
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# For tensorboard
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")  # For fake images
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")  # For real images
step = 0

# Prepare models for training
generator.train()
discriminator.train()

for epoch in range(epochs):
    for batch_idx, (real_image, _) in enumerate(loader):
        batch_size = real_image.shape[0]

        ## Fetching real image and generating FAKE image
        # Real image represents the actual MNIST image
        real_image = real_image.view(-1, 28 * 28).to(device)
        # Noise represents a random matrix of noise used by the generator as input
        noise = torch.randn(batch_size, noise_dimension).to(device)
        fake_image = generator(noise)

        ## Train discriminator, maximize (log(D(real)) + log(1-D(G(z))))
        # log(D(real)) part
        discriminator_real = discriminator(real_image).view(-1)
        loss_discriminator_real = criterion(discriminator_real, torch.ones_like(discriminator_real)) # Discriminator should associate real image with  1
        # log(1-D(G(z)) part
        discriminator_fake = discriminator(fake_image).view(-1)
        loss_discriminator_fake = criterion(discriminator_fake, torch.zeros_like(discriminator_fake)) # Discriminator should associate fake image with  0
        # Combined loss
        loss_discriminator = (loss_discriminator_real + loss_discriminator_fake) / 2

        discriminator.zero_grad()
        # loss.backward() sets the grad attribute of all tensors with requires_grad=True in the computational graph
        loss_discriminator.backward(retain_graph=True)  # retrain_graph so that what was used in this pass is not cleared from cache, ex: fake_image
        opt_discriminator.step()

        ## Train generator, minimize log(1-D(G(z))) OR maximize log(D(G(z)))
        # log(D(G(z)) part
        discriminator_fake = discriminator(fake_image).view(-1)
        loss_generator = criterion(discriminator_fake, torch.ones_like(discriminator_fake)) # Generator wants Discriminator to associate fake image with  1

        generator.zero_grad()
        loss_generator.backward(retain_graph=True)  # retrain_graph so that what was used in this pass is not cleared from cache, ex: fake_image
        opt_generator.step()

        # Tensorboard code
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {loss_discriminator:.4f}, loss G: {loss_generator:.4f}"
            )

            with torch.no_grad():
                fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                data = real_image.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Real Images", img_grid_real, global_step=step
                )
                step += 1
