"""
mnist_gan.py
Generates handwritten digits with a Generative Adversarial Network (GAN) using PyTorch.
Author: Sansu (AI Student)
"""
import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torchvision
from PIL import Image

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader


def get_mnist_dataloader(batch_size, data_dir="data"):
    """
    Downloads and loads the MNIST dataset with transforms for GAN training.
    """
    print("\U0001F4E5 Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"\u2705 MNIST loaded with {len(dataset)} samples.")
    return dataloader


def preview_real_data(dataloader):
    """Displays a grid of real MNIST digits."""
    real_imgs, _ = next(iter(dataloader))
    grid = torchvision.utils.make_grid(real_imgs[:16], nrow=4, normalize=True)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Real MNIST Samples")
    plt.axis("off")
    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description="Train a GAN on MNIST")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--out_dir', type=str, default='images')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(z.size(0), 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def train():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dataloader = get_mnist_dataloader(args.batch_size)
    preview_real_data(dataloader)

    device = torch.device(args.device)
    G = Generator(args.latent_dim).to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, args.latent_dim, device=device)

    for epoch in range(1, args.epochs + 1):
        for batch_idx, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = criterion(D(real_imgs), valid)
            z = torch.randn(batch_size, args.latent_dim, device=device)
            gen_imgs = G(z)
            fake_loss = criterion(D(gen_imgs.detach()), fake)
            loss_D = real_loss + fake_loss
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            loss_G = criterion(D(gen_imgs), valid)
            loss_G.backward()
            optimizer_G.step()

        print(f"[Epoch {epoch}/{args.epochs}] Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}")

        with torch.no_grad():
            fake_imgs = G(fixed_noise).detach().cpu()
            save_image(fake_imgs, f"{args.out_dir}/{epoch:03d}.png", nrow=8, normalize=True)

    torch.save(G.state_dict(), 'generator.pth')
    torch.save(D.state_dict(), 'discriminator.pth')
    print("Training finished. Models saved as generator.pth and discriminator.pth.")

    final_img_path = f"{args.out_dir}/{args.epochs:03d}.png"
    if os.path.exists(final_img_path):
        image = Image.open(final_img_path)
        plt.imshow(image)
        plt.title("Final Generated Digits")
        plt.axis("off")
        plt.show()

    return args


import imageio

import imageio

def create_gif(image_folder='images', output_path='generated_digits.gif', duration=0.3):
    """
    Creates a GIF from PNG images in a folder.
    Args:
        image_folder (str): Directory containing epoch images.
        output_path (str): File path for the output GIF.
        duration (float): Time between frames in seconds.
    """
    print("🎞️ Creating GIF...")
    images = []
    files = sorted(os.listdir(image_folder))
    for file_name in files:
        if file_name.endswith('.png'):
            file_path = os.path.join(image_folder, file_name)
            images.append(imageio.v2.imread(file_path))
    imageio.mimsave(output_path, images, duration=duration)
    print(f"✅ GIF saved to {output_path}")



if __name__ == '__main__':
    args = train()  # capture args returned
    create_gif(args.out_dir, "training_progress.gif")



