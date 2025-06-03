from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
import os


def load_data(batch_size=64, validation_split=0.2, data_dir="../data"):
    """
    Load dataloaders for train, test, and validation datasets.
    :param batch_size: the batch size for training
    :param validation_split: the fraction of training data to use for validation
    :param data_dir: directory to save/load the MNIST data
    :return: train, validation, and test DataLoaders
    """
    # Check if the data is already downloaded
    if not os.path.exists(os.path.join(data_dir, "MNIST")):
        print(f"Data not found in {data_dir}, downloading...")
        download = True
    else:
        download = False

    # Load the MNIST dataset
    train_data = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=ToTensor()
    )

    # Split the training data into training and validation datasets
    train_data, valid_data = train_test_split(train_data, test_size=validation_split)

    # Create DataLoaders for train, validation, and test data
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
