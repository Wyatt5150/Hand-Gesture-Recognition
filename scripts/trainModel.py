'''
Ymai's Notes
    Renamed SignLanguageMNISTDataModule to DataModule
    see notes in dataModule.py for more information 
'''

"""
This script trains a Sign Language gesture recognition model using PyTorch Lightning.
It supports data loading, model checkpointing, early stopping, and automatic GPU utilization.
"""

import os
import sys
import glob  # Import glob to find the latest checkpoint

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import pytorch_lightning as pl
import torch  # Import torch for saving model weights
from models.model import SignLanguageCNN
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from scripts.dataModule import DataModule

# Check and print GPU availability once at the beginning of the script
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

class TrainModel:
    """
    A class to manage the training process of the SignLanguageCNN model using PyTorch Lightning.

    Attributes:
        data_module (DataModule): Instance of the data module for loading and augmenting data.
        model (SignLanguageCNN): Instance of the gesture recognition model.
        checkpoint_dir (str): Directory path for saving model checkpoints.
    """
    def __init__(self):
        """
        Initializes the TrainModel class, setting up the data module, model, and checkpoint directory.
        """
        self.data_module = DataModule(
            train_csv='sign_mnist_train.csv',
            val_csv='sign_mnist_test.csv',
            test_csv='sign_mnist_test.csv',
            batch_size=128,
            apply_augmentation=True  # Set to True for training
        )
        self.model = SignLanguageCNN(num_classes=24, learning_rate=0.0001)
        self.checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'utils', 'checkpoints')

    def get_latest_checkpoint(self):
        """
        Finds and returns the most recent model checkpoint.

        Returns:
            str or None: Path to the latest checkpoint file, or None if no checkpoints are found.
        """
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, '*.ckpt'))
        if not checkpoints:
            return None
        return max(checkpoints, key=os.path.getctime)  # Return the latest checkpoint

    def train(self):
        """
         Trains the SignLanguageCNN model, handling early stopping, model checkpointing, and logging.

         - Initializes the data module and callbacks for early stopping and checkpointing.
         - Sets up a PyTorch Lightning trainer with GPU support if available.
         - Resumes training from the latest checkpoint, if available.
         - Evaluates the model on a test set after training and saves the model's weights.

         This method does not return any values.
         """
        # Call setup to initialize datasets
        self.data_module.setup()

        # Setup callbacks for early stopping and model checkpointing
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=self.checkpoint_dir,
            filename="best-checkpoint",
            save_top_k=1,
            mode="min"
        )

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=7,
            verbose=True,
            mode="min"
        )

        logger = pl.loggers.TensorBoardLogger(
            save_dir=os.path.join(os.path.dirname(__file__), '..', 'utils', 'lightning_logs'),
            name="gesture-recognition"
        )

        # Initialize the trainer
        trainer = pl.Trainer(
            max_epochs=20,  # Set your desired number of epochs
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=logger,
            accelerator='auto',  # Automatically use GPU if available
            devices='auto'       # Automatically use as many devices as possible
        )

        # Load model from the latest checkpoint if it exists
        latest_checkpoint = self.get_latest_checkpoint()
        if latest_checkpoint:
            print("Loading existing model from checkpoint...")
            self.model = SignLanguageCNN.load_from_checkpoint(latest_checkpoint, strict=False)

        # Start training the model
        trainer.fit(self.model, train_dataloaders=self.data_module.train_dataloader(), val_dataloaders=self.data_module.val_dataloader())

        # Test the model after training using the test dataloader
        trainer.test(self.model, dataloaders=self.data_module.test_dataloader())

        # Save the model weights to a .pth file
        torch.save(self.model.state_dict(), os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pth'))

if __name__ == "__main__":
    training_instance = TrainModel()
    training_instance.train()
