import sys
import os
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the path where images are stored and where the CSV will be saved
# image_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
# csv_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

image_dir = '../images'
csv_dir = '../data'


# Paths for the split CSV files
train_csv_path = os.path.join(csv_dir, 'custom_sign_language_train.csv')
val_csv_path = os.path.join(csv_dir, 'custom_sign_language_val.csv')
test_csv_path = os.path.join(csv_dir, 'custom_sign_language_test.csv')

# Initialize an empty list to hold the image data and labels
image_data = []
labels = []

# Define a label mapping for letters to numbers
label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
                 'L': 10, 'M': 11, 'N': 12, 'O': 13, 'P': 14, 'Q': 15, 'R': 16, 'S': 17, 'T': 18,
                 'U': 19, 'V': 20, 'W': 21, 'X': 22, 'Y': 23}

# Loop through each subdirectory in the images directory (each subdirectory corresponds to a gesture label)
for gesture_label in os.listdir(image_dir):
    gesture_dir = os.path.join(image_dir, gesture_label.upper())  # Ensure label is in uppercase

    # Check if the directory exists and is not a file
    if os.path.isdir(gesture_dir) and gesture_label.upper() in label_mapping:
        numeric_label = label_mapping[gesture_label.upper()]

        # Loop through each image in the gesture's directory
        for img_file in os.listdir(gesture_dir):
            img_path = os.path.join(gesture_dir, img_file)

            # Read the image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # Skip if the image can't be read

            # Resize the image to 28x28 (if it's not already)
            img_resized = cv2.resize(img, (28, 28))

            # Flatten the image into a 1D array
            img_flattened = img_resized.flatten()

            # Add the image data and the corresponding numeric label to the list
            image_data.append(img_flattened)
            labels.append(numeric_label)

# Convert the lists to a DataFrame
df = pd.DataFrame(image_data)
df.insert(0, 'label', labels)  # Insert the numeric label as the first column

# Split the data into train (70%), validation (15%), and test (15%)
train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42, stratify=labels)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['label'])

# Save the splits to CSV files
train_data.to_csv(train_csv_path, index=False)
val_data.to_csv(val_csv_path, index=False)
test_data.to_csv(test_csv_path, index=False)

print(f"Training CSV saved at {train_csv_path}")
print(f"Validation CSV saved at {val_csv_path}")
print(f"Test CSV saved at {test_csv_path}")
