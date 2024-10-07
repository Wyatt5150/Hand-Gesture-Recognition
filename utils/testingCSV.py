import sys
import os
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch

# Fix for OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Define the base directory dynamically using os.path
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# Paths to the CSV files
train_csv_path = os.path.join(data_dir, 'custom_sign_language_train.csv')
test_csv_path = os.path.join(data_dir, 'custom_sign_language_val.csv')

# Read the CSV files
train_data = pd.read_csv(train_csv_path)
test_data = pd.read_csv(test_csv_path)

# Display basic information about the training data
print("Training Data Info:")
print(train_data.info())
print("\nTraining Data Head:")
print(train_data.head())

# Display basic information about the testing data
print("\nTesting Data Info:")
print(test_data.info())
print("\nTesting Data Head:")
print(test_data.head())

# Check unique labels in training data
print("\nUnique Labels in Training Data:")
print(train_data.iloc[:, 0].unique())  # Assuming labels are in the first column

# Dictionary mapping labels (0-24) to letters (excluding 'J' and 'Z')
label_to_letter = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
    10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T',
    19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

def visualize_random_images(data, num_images=5, as_tensor=False):
    """
    Visualize random images from the dataset.

    Parameters:
        data (DataFrame): The dataset containing image data and labels.
        num_images (int): The number of random images to visualize (default is 5).
        as_tensor (bool): If True, converts images to tensors and prints their shapes (default is False).
    """
    plt.figure(figsize=(10, 10))

    # Randomly select indices from the data
    random_indices = np.random.choice(len(data), num_images, replace=False)

    for i, idx in enumerate(random_indices):
        plt.subplot(1, num_images, i + 1)
        img_array = data.iloc[idx, 1:].values.astype(np.uint8).reshape(28, 28)  # Adjust according to your data
        label = data.iloc[idx, 0]
        letter = label_to_letter.get(label, 'Unknown')  # Get the associated letter

        if as_tensor:
            # Convert to tensor (optional)
            img_tensor = torch.tensor(img_array).float().unsqueeze(0)
            print(f"Tensor shape for label {label}: {img_tensor.shape}")

        # Display the image and its associated letter and label
        plt.imshow(img_array, cmap='gray')
        plt.title(f'Label: {label}\nLetter: {letter}')  # Display both label and letter
        plt.axis('off')
    plt.show()

# Visualize random images from training data (Set as_tensor=True if needed)
visualize_random_images(train_data, num_images=5, as_tensor=False)
