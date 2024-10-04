import os  # Import the os module
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping  # Import EarlyStopping
import torch
import sys

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your model and data module here
from models.model import SignLanguageCNN
from preProcessing import SignLanguageMNISTDataModule

class SignLanguageTraining:
    """
    A class for training a Sign Language recognition model using PyTorch Lightning.

    This class sets up the data module, model, logger, and callbacks for training,
    validation, and testing.

    Example:
    >>> trainer = SignLanguageTraining()
    >>> trainer.train()
    """

    def __init__(self):
        """
        Initializes the SignLanguageTraining class.

        Sets up the logger, callbacks, model, and data module.
        """
        # Define logger and checkpoint callback
        self.logger = TensorBoardLogger(
            os.path.join("..", "tb_logs"),  # Adjusted path to create tb_logs in the root directory
            name="sign_language_mnist_model"
        )

        self.checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode='min',
            filename='{epoch}-{val_loss:.2f}'
        )

        self.lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # Add EarlyStopping callback
        self.early_stopping_callback = EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=3,  # Number of epochs with no improvement after which training will be stopped
            verbose=True,
            mode='min'  # Minimize the validation loss
        )

        # Initialize data module and model
        self.data_module = SignLanguageMNISTDataModule()
        self.model = SignLanguageCNN()

        # Use all available CPU cores for training
        self.devices = "auto"  # Automatically use available devices


    def train(self):
        """
        Trains the Sign Language recognition model.

        Initializes the PyTorch Lightning trainer and fits the model to the data.

        Returns:
        - None

        Example:
        >>> trainer = SignLanguageTraining()
        >>> trainer.train()
        """
        # Initialize trainer
        trainer = Trainer(
            max_epochs=30,
            logger=self.logger,
            callbacks=[self.checkpoint_callback, self.lr_monitor, self.early_stopping_callback],
            devices=12,
            strategy="auto",  # Set strategy to "auto"
        )

        # Train the model
        trainer.fit(self.model, self.data_module)


        # Test the model
        trainer.test(self.model, datamodule=self.data_module)

        # Create 'models' directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Save the model weights
        torch.save(self.model.state_dict(), os.path.join(models_dir, 'model.pth'))
        print("Model weights saved to models/model.pth")


if __name__ == "__main__":
    training_instance = SignLanguageTraining()
    training_instance.train()
