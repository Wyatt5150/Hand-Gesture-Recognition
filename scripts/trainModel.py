import os
import sys
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytorch_lightning as pl
import torch  # Import torch for saving model weights
from models.model import SignLanguageCNN
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from scripts.preProcessing import SignLanguageMNISTDataModule

class TrainModel:
    def __init__(self):
        self.data_module = SignLanguageMNISTDataModule(
            train_csv=os.path.join(os.path.dirname(__file__), '..', 'data', 'custom_sign_language_train.csv'),
            val_csv=os.path.join(os.path.dirname(__file__), '..', 'data', 'custom_sign_language_val.csv'),
            test_csv=os.path.join(os.path.dirname(__file__), '..', 'data', 'custom_sign_language_test.csv'),
            batch_size=128
        )
        self.model = SignLanguageCNN(num_classes=24, learning_rate=0.0001)
        self.checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'checkpoints', 'best-checkpoint.ckpt')

    def train(self):
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(os.path.dirname(__file__), '..', 'utils', 'checkpoints'),
            filename="best-checkpoint",
            save_top_k=1,
            mode="min"
        )

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=3,
            verbose=True,
            mode="min"
        )

        logger = pl.loggers.TensorBoardLogger(
            save_dir=os.path.join(os.path.dirname(__file__), '..', 'utils', 'lightning_logs'),
            name="gesture-recognition"
        )

        trainer = pl.Trainer(
            max_epochs=10,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=logger,
            accelerator='auto',
            devices='auto'
        )

        if os.path.exists(self.checkpoint_path):
            print("Loading existing model from checkpoint...")
            self.model = SignLanguageCNN.load_from_checkpoint(self.checkpoint_path)

        trainer.fit(self.model, train_dataloaders=self.data_module.train_dataloader(), val_dataloaders=self.data_module.val_dataloader())
        trainer.test(self.model, dataloaders=self.data_module.test_dataloader())

        # Save the model weights to a .pth file
        torch.save(self.model.state_dict(), os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pth'))

if __name__ == "__main__":
    training_instance = TrainModel()
    training_instance.train()
