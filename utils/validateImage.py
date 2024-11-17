import os
import sys
import torch
from PIL import Image
from torchvision import transforms
import random

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import SignLanguageCNN  # Ensure this import is correct

# Label-to-letter mapping
label_to_letter = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
    10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T',
    19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

def load_and_predict(image_path):
    """
    Load a pre-trained model and predict the gesture represented by the given image.

    Parameters:
        image_path (str): The absolute path to the image file to be predicted.

    Returns:
        None
    """
    # Load the model
    model = SignLanguageCNN()

    # Use absolute path for model.pth
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Transform the image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    # Open and transform the image
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        predicted_label = predicted.item()

        # Get the associated letter from the predicted label
        predicted_letter = label_to_letter.get(predicted_label, 'Unknown')

        print(f'Predicted letter for {os.path.basename(image_path)}: {predicted_letter}')

def get_random_image(images_dir):
    """
    Randomly select an image from a random subdirectory.

    Parameters:
        images_dir (str): The directory containing subdirectories with gesture images.

    Returns:
        str: The path to a randomly selected image file.
    """
    # Get all subdirectories (letters)
    subdirectories = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]

    if not subdirectories:
        print("No subdirectories found.")
        return None

    # Select a random subdirectory (letter)
    random_subdir = random.choice(subdirectories)
    subdir_path = os.path.join(images_dir, random_subdir)

    # Get all .png files in the selected subdirectory
    png_files = [f for f in os.listdir(subdir_path) if f.endswith('.png')]

    if not png_files:
        print(f"No .png files found in {random_subdir}.")
        return None

    # Select a random .png file
    random_image = random.choice(png_files)

    return os.path.join(subdir_path, random_image)

# Use absolute path for images folder
images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')

# Get a random image path and predict its label
random_image_path = get_random_image(images_dir)

if random_image_path:
    load_and_predict(random_image_path)
else:
    print("No image found to predict.")
