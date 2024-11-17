"""
This script trains a Sign Language gesture recognition model
"""

import os
import sys
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.dataModule import DataModule
from scripts.model import SignLanguageCNN

# Model Parameters
dataset = 'sign_mnist'
epochs = 20
batch_size = 128
model_file_name = 'model'


if __name__ == "__main__":
    # loads data and trains model based on values specified above
    data = DataModule(dataset)
    model = SignLanguageCNN()
    model.train(data, batch_size=batch_size, epochs=epochs, model_name = model_file_name)