import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

def get_mnist_dataloader(batch_size, data_dir="data"):
    """
    Downloads and loads the MNIST dataset with transforms for GAN training.
    Args:
        batch_size (int): Number of samples per batch.
        data_dir (str): Directory where MNIST will be downloaded/stored.

    Returns:
        torch.utils.data.DataLoader: PyTorch dataloader with MNIST images.
    """
    print("📥 Loading MNIST dataset...")

    # Define transform: convert to tensor and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Load the MNIST dataset with automatic downloading
    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True
    )

    # Wrap it in a DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    print(f"✅ MNIST loaded with {len(dataset)} samples.")
    return dataloader
