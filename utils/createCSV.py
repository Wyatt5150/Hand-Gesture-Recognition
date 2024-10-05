import os
import cv2
import numpy as np
import pandas as pd

# Define the path where images are stored and where the CSV will be saved
image_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'custom_sign_language_data.csv')

# Initialize an empty list to hold the image data and labels
image_data = []
labels = []

# Loop through each subdirectory in the images directory (each subdirectory corresponds to a gesture label)
for gesture_label in os.listdir(image_dir):
    gesture_dir = os.path.join(image_dir, gesture_label)

    # Check if the directory exists and is not a file
    if os.path.isdir(gesture_dir):
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

            # Add the image data and the corresponding label to the list
            image_data.append(img_flattened)
            labels.append(gesture_label)

# Convert the lists to a DataFrame
df = pd.DataFrame(image_data)
df.insert(0, 'label', labels)  # Insert the label as the first column

# Save the DataFrame to a CSV file
df.to_csv(csv_path, index=False)
print(f"CSV file saved at {csv_path}")
