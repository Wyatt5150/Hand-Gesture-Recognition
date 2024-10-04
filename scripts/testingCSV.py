import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Define the base directory dynamically using os.path
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# Paths to the CSV files
train_csv_path = os.path.join(data_dir, 'sign_mnist_train.csv')
test_csv_path = os.path.join(data_dir, 'sign_mnist_test.csv')

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
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'Skip',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}


# Function to visualize random images
def visualize_random_images(data, num_images=5):
    plt.figure(figsize=(10, 10))

    # Randomly select indices from the data
    random_indices = np.random.choice(len(data), num_images, replace=False)

    for i, idx in enumerate(random_indices):
        plt.subplot(1, num_images, i + 1)
        img_array = data.iloc[idx, 1:].values.astype(np.uint8).reshape(28, 28)  # Adjust according to your data
        label = data.iloc[idx, 0]
        letter = label_to_letter.get(label, 'Unknown')  # Get the associated letter

        # Display the image and its associated letter and label
        plt.imshow(img_array, cmap='gray')
        plt.title(f'Label: {label}\nLetter: {letter}')  # Display both label and letter
        plt.axis('off')
    plt.show()


# Visualize random images from training data
visualize_random_images(train_data, num_images=5)
