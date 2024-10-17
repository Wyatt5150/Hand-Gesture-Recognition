'''
Ymai's Notes
    this seems like a worse version of dataModule.py and i didnt find anything that uses it
    why does this exist
'''

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pytorch_lightning import LightningDataModule

# Dataset class
class SignLanguageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Initialize the SignLanguageDataset.

        Parameters:
            csv_file (str): Path to the CSV file containing image data and labels.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where `image` is a tensor of the image and `label` is the corresponding label tensor.
        """
        row = self.data.iloc[idx]
        image_data = row.iloc[1:].values.astype(np.uint8)  # The rest are pixel values
        image = image_data.reshape(28, 28)
        label = int(row.iloc[0])  # Ensure the label is read as an integer
        if self.transform:
            image = self.transform(image)  # This will already be a tensor if a transform is applied

        # Ensure that `image` and `label` are tensors by using `.clone().detach()` if they already are
        return image.clone().detach(), torch.tensor(label)

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

# DataModule class
class SignLanguageMNISTDataModule(LightningDataModule):
    def __init__(self, train_csv, val_csv, test_csv, batch_size=32):
        """
        Initialize the SignLanguageMNISTDataModule.

        Parameters:
            train_csv (str): Path to the training CSV file.
            val_csv (str): Path to the validation CSV file.
            test_csv (str): Path to the testing CSV file.
            batch_size (int): Batch size for the DataLoader (default: 32).
        """
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size

        # Define the transformations
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and std deviation
        ])
        self.val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def setup(self, stage=None):
        """
        Setup the datasets for training, validation, and testing.

        Parameters:
            stage (str, optional): The stage for which to setup the datasets. Can be 'fit' for training/validation
                                   or 'test' for testing. If None, sets up all datasets.
        """
        # Initialize datasets for training, validation, and testing
        if stage == 'fit' or stage is None:
            self.train_dataset = SignLanguageDataset(self.train_csv, transform=self.train_transforms)
            self.val_dataset = SignLanguageDataset(self.val_csv, transform=self.val_transforms)

        if stage == 'test' or stage is None:
            self.test_dataset = SignLanguageDataset(self.test_csv, transform=self.test_transforms)

    def train_dataloader(self):
        """
        Create the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        self.setup('fit')  # Ensure dataset is set up before returning the dataloader
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """
        Create the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        self.setup('fit')
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        """
        Create the DataLoader for the testing dataset.

        Returns:
            DataLoader: DataLoader for the testing dataset.
        """
        self.setup('test')
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
