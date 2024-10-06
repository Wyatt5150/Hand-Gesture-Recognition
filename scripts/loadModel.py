import os
import sys
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.model import SignLanguageCNN  # Assuming this is your model

def load_model_version(version_number, checkpoint_name=None):
    """
    Load a specific version of the trained model.

    Parameters:
        version_number (int): The version number of the model to load.
        checkpoint_name (str, optional): Specific checkpoint name to load.
            If None, the first available checkpoint will be loaded.

    Returns:
        model (SignLanguageCNN): The loaded model with weights from the specified version.

    Raises:
        FileNotFoundError: If the checkpoint directory or the specified checkpoint file does not exist.
    """
    # Construct the path to the checkpoint directory
    checkpoint_path = os.path.join(
        os.path.dirname(__file__),  # Current directory (scripts)
        '..', 'tb_logs', 'sign_language_mnist_model', f'version_{version_number}', 'checkpoints'
    )

    # Check if the checkpoint path exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory for version {version_number} not found at: {checkpoint_path}")

    # Find all checkpoint files in the directory
    checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')]

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_path}")

    # If no specific checkpoint name is provided, load the first available one
    if checkpoint_name is None:
        checkpoint_file = checkpoint_files[0]  # You could also sort and pick the latest if needed
    else:
        # Check if the specified checkpoint exists
        if checkpoint_name not in checkpoint_files:
            raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found in: {checkpoint_path}")
        checkpoint_file = checkpoint_name

    # Full path to the checkpoint file
    model_path = os.path.join(checkpoint_path, checkpoint_file)

    # Initialize model
    model = SignLanguageCNN()

    # Load the model state dict from the checkpoint
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))["state_dict"], strict=False)
    model.eval()  # Set the model to evaluation mode

    print(f'Model loaded from version_{version_number}, checkpoint: {checkpoint_file}')
    return model

def main():
    """
    Main function to load a specific version of the trained model.

    This function calls `load_model_version` with a specified version and
    optionally a checkpoint name. It can be extended for further tasks
    such as testing or inference.
    """
    # Load a specific version of the model
    version_to_load = 27  # Change this to the version you want to load
    checkpoint_to_load = None  # Optionally specify a checkpoint file name, or leave None to load the first one
    model = load_model_version(version_to_load, checkpoint_to_load)

    # Now you can use the `model` for further tasks like testing, inference, etc.
    # Example:
    # - Use it for gesture recognition
    # - Run inference on some data
    # - Continue training from this checkpoint if needed

if __name__ == "__main__":
    main()
