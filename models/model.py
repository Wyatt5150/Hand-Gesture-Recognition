import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule

class SignLanguageCNN(LightningModule):
    def __init__(self, num_classes=24, learning_rate=0.0001):
        """
        Initializes the SignLanguageCNN model with increased features.

        Args:
            num_classes (int): Number of output classes (default is 24).
            learning_rate (float): Learning rate for the optimizer (default is 0.0001).
        """
        super(SignLanguageCNN, self).__init__()
        self.learning_rate = learning_rate

        # Convolutional layers with batch normalization to increase accuracy
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(1024 * 1 * 1, 256)  # Adjust this input size based on your output from conv layers
        self.fc_middle = nn.Linear(256, 128)     # Increase the size of fully connected layers for more learning capacity
        self.fc2 = nn.Linear(128, num_classes)   # 24 output classes

        # Dropout for regularization
        self.dropout = nn.Dropout(0.4)

        # Initialize accuracy metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits from the final layer.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten dynamically based on actual size
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc_middle(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (tuple): Tuple containing input data and labels.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value for the current batch.
        """
        x, y = batch
        logits = self(x)
        y = y.long()
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        acc = self.train_accuracy(logits, y)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch (tuple): Tuple containing input data and labels.
            batch_idx (int): Batch index.

        Returns:
            dict: Dictionary containing validation loss and accuracy.
        """
        x, y = batch
        logits = self(x)
        y = y.long()
        loss = F.cross_entropy(logits, y)
        val_acc = self.val_accuracy(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)
        return {'val_loss': loss, 'val_acc': val_acc}

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.

        Args:
            batch (tuple): Tuple containing input data and labels.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value for the test batch.
        """
        x, y = batch
        logits = self(x)
        y = y.long()
        loss = F.cross_entropy(logits, y)
        test_acc = self.test_accuracy(logits, y)
        self.log('test_loss', loss)
        self.log('test_acc', test_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configures optimizers and learning rate schedulers.

        Returns:
            list: List of optimizers.
            list: List of learning rate schedulers.
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5),
            'monitor': 'val_loss',
        }
        return [optimizer], [scheduler]
