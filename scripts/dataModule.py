import os  # Import os for file path management
import pytorch_lightning as pl
import torch  # Import torch here for tensor operations
from torch.utils.data import DataLoader, Dataset  # Import Dataset here
import torchvision.transforms as transforms
from torchvision.transforms import Lambda
import pandas as pd
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Initializes the CustomDataset.

        Args:
            csv_file (str): Path to the CSV file containing labels and pixel values for images.
            transform (callable, optional): Optional transformation to be applied on an image sample.
        """
        self.data_frame = pd.read_csv(csv_file)  # Load the CSV file
        self.transform = transform

    def __len__(self):
        """
          Returns the number of samples in the dataset.

          Returns:
              int: Number of samples in the dataset.
          """
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
          Retrieves an image and label pair for a given index.

          Args:
              idx (int): Index of the sample to retrieve.

          Returns:
              tuple: A tuple containing the transformed image (as a tensor) and its corresponding label.
          """
        label = self.data_frame.iloc[idx, 0]  # Get the label from the first column
        pixel_values = self.data_frame.iloc[idx, 1:].values.astype('float32')  # Get pixel values as float

        # Reshape the pixel values into the correct image dimensions (e.g., 28x28)
        image = pixel_values.reshape(28, 28)  # Adjust dimensions based on your dataset
        image = Image.fromarray(image)  # Convert to PIL Image

        # Ensure image is in the right mode
        if image.mode != 'L':
            image = image.convert('L')  # Convert to grayscale if not already

        if self.transform:
            image = self.transform(image)  # Apply any transformations

        return image, label


# Define the cutout transformation function
def cutout_transform(img, size=8):
    """
    Applies cutout augmentation to an image.

    Args:
        img (PIL.Image or torch.Tensor): Image to which cutout is applied.
        size (int): Size of the square patch to cut out.

    Returns:
        torch.Tensor: Transformed image with a cutout patch.
    """
    return cutout_fn(img, size)


# Cutout function that modifies the image
def cutout_fn(img, size=8):
    """
    Zeroes out a randomly located square patch in the image for cutout augmentation.

    Args:
        img (torch.Tensor): Input image tensor.
        size (int): Size of the square patch to be cut out.

    Returns:
        torch.Tensor: Image tensor with a square patch zeroed out.
    """
    h, w = img.size(1), img.size(2)
    y = torch.randint(low=0, high=h, size=(1,)).item()
    x = torch.randint(low=0, high=w, size=(1,)).item()

    y1 = max(0, y - size // 2)
    y2 = min(h, y + size // 2)
    x1 = max(0, x - size // 2)
    x2 = min(w, x + size // 2)

    img[:, y1:y2, x1:x2] = 0  # Zero out the region
    return img


class SignLanguageMNISTDataModule(pl.LightningDataModule):
    def __init__(self, train_csv, val_csv, test_csv, batch_size, apply_augmentation=True):
        """
        Initializes the SignLanguageMNISTDataModule.

        Args:
            train_csv (str): Path to the training CSV file.
            val_csv (str): Path to the validation CSV file.
            test_csv (str): Path to the test CSV file.
            batch_size (int): Batch size for data loading.
            apply_augmentation (bool): Whether to apply data augmentations to the training set.
        """
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.apply_augmentation = apply_augmentation  # Flag to control augmentation

        # Data augmentations for the training set v1 (previous version for reference)
        # self.data_transforms = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(10),
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #     transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5], std=[0.5])
        # ])

        # Data augmentation refinement to increase accuracy set v2 (current version)
        self.data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(28, scale=(0.6, 1.0)),  # More aggressive cropping
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            # Affine transformations
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            # Replaced lambda with a named function for cutout
            Lambda(cutout_transform)
        ])

        # Define transforms for validation and test (no augmentations)
        self.test_val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def prepare_data(self):
        """
        Prepares data if necessary. This function is a placeholder for any required data preparation steps.
        """
        # Prepare your data here if necessary
        pass

    def setup(self, stage=None):
        """
        Sets up the datasets for training, validation, and testing.

        Args:
            stage (str, optional): The stage of the setup ('fit', 'test', etc.). Defaults to None.
        """
        # Apply augmentations only to the training dataset
        self.train_dataset = CustomDataset(os.path.join('..', 'data', self.train_csv),
                                           transform=self.data_transforms if self.apply_augmentation else self.test_val_transforms)
        self.val_dataset = CustomDataset(os.path.join('..', 'data', self.val_csv),
                                         transform=self.test_val_transforms)  # No augmentations
        self.test_dataset = CustomDataset(os.path.join('..', 'data', self.test_csv),
                                          transform=self.test_val_transforms)  # No augmentations

    def train_dataloader(self):
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=11,
                          persistent_workers=True)

    def val_dataloader(self):
        """
         Returns the training DataLoader.

         Returns:
             DataLoader: DataLoader for the training dataset.
         """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=11, persistent_workers=True)

    def test_dataloader(self):
        """
        Returns the test DataLoader.

        Returns:
            DataLoader: DataLoader for the test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=11, persistent_workers=True)