import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torchvision import transforms


class SignLanguageMNISTDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for loading the Sign Language MNIST dataset.

    This module handles loading, preprocessing, and splitting the dataset into training,
    validation, and test sets.

    Parameters:
    - data_dir (str): The directory where the dataset CSV files are located. Default is
      'C:\\Users\\Lopez\\Documents\\Hand-Gesture-Recognition\\data'.
    - batch_size (int): The number of samples per batch. Default is 64.

    Example:
    >>> data_module = SignLanguageMNISTDataModule()
    >>> data_module.prepare_data()
    >>> data_module.setup()
    """

    def __init__(self, data_dir='C:\\Users\\Lopez\\Documents\\Hand-Gesture-Recognition\\data', batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def prepare_data(self):
        """No download needed as data is already in data_dir."""
        pass

    def setup(self, stage=None):
        """
        Set up the dataset for training, validation, and testing.

        Parameters:
        - stage (str): The stage of training ('fit', 'test', etc.). If None, setups all stages.
        """
        train_csv = f'{self.data_dir}\\sign_mnist_train.csv'
        if stage == 'fit' or stage is None:
            train_csv = os.path.join(self.data_dir, 'sign_mnist_train.csv')
            print(f"Loading training data from: {train_csv}")  # Debug print
            train_dataset = SignLanguageMNIST(csv_file=train_csv, transform=self.transform)

            num_train = int(len(train_dataset) * 0.8)
            num_val = len(train_dataset) - num_train
            self.train_dataset, self.val_dataset = random_split(train_dataset, [num_train, num_val])

        if stage == 'test' or stage is None:
            test_csv = os.path.join(self.data_dir, 'sign_mnist_test.csv')
            print(f"Loading testing data from: {test_csv}")  # Debug print
            self.test_dataset = SignLanguageMNIST(csv_file=test_csv, transform=self.transform)

    def train_dataloader(self):
        """Return the DataLoader for training data."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Return the DataLoader for validation data."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """Return the DataLoader for test data."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class SignLanguageMNIST(Dataset):
    """
    A custom Dataset class for the Sign Language MNIST dataset.

    This class loads image data from a CSV file and returns images and their corresponding labels.

    Parameters:
    - csv_file (str): Path to the CSV file containing the dataset.
    - transform (callable, optional): A function/transform to apply to the images.

    Example:
    >>> dataset = SignLanguageMNIST(csv_file='path/to/sign_mnist_train.csv', transform=my_transform)
    >>> image, label = dataset[0]  # Get the first image and its label
    """

    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - Tuple[Tensor, int]: A tuple containing the transformed image tensor and its label.
        """
        image = self.data_frame.iloc[idx, 1:].values.astype(np.uint8).reshape(28, 28)
        image = Image.fromarray(image, 'L')
        label = int(self.data_frame.iloc[idx, 0])

        if self.transform:
            image = self.transform(image)

        return image, label
