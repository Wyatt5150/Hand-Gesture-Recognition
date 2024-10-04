import os
import sys
import torch
from PIL import Image
from torchvision import transforms

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import SignLanguageCNN  # Ensure this import is correct

def load_and_predict(image_path):
    # Load the model
    model = SignLanguageCNN()

    # Use absolute path for model.pth
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pth')
    model.load_state_dict(torch.load(model_path, weights_only=True))
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
        print(f'Predicted label for {os.path.basename(image_path)}: {predicted.item()}')

# Use absolute path for images folder
images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')

# Loop through each letter image
for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']:
    image_path = os.path.join(images_dir, f'{letter}.png')  # Absolute path to each image
    load_and_predict(image_path)
