import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

class DataModule():
    """
        Class to manage dataset for training
    """
    def __init__(self, dataset:str):
        """
        Initializes the DataModule using a specified dataset

        Parameters:
            dataset (str): name of dataset being used

        Returns:
            none
        """

        # load data sets
        curDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        print(os.path.join(curDir, 'data','train', dataset+'_train.csv'))
        train_df=pd.read_csv(os.path.join(curDir, 'data','train', dataset+'_train.csv'))
        test_df=pd.read_csv(os.path.join(curDir, 'data','test', dataset+'_test.csv'))

        # separate labels from the rest of the data
        self.train_labels = train_df['label']
        self.train_data = train_df.drop(['label'],axis=1)
        self.train_data = self.train_data.values.reshape(-1,28,28,1)

        self.test_labels = test_df['label']
        self.test_data = test_df.drop(['label'],axis=1)
        self.test_data = self.test_data.values.reshape(-1,28,28,1)
        self.test_data = self.test_data/255

        lb=LabelBinarizer()
        self.train_labels=lb.fit_transform(self.train_labels)
        self.test_labels=lb.fit_transform(self.test_labels)

        # distortions to data to improve training
        self.train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range = 10,
                                  height_shift_range=0.2,
                                  width_shift_range=0.2,
                                  shear_range=0,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

    def get_test_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns data for testing

        Parameters:
            none

        Returns:
            tuple (labels, data) : contains test dataset
        """
        return (self.test_labels, self.test_data)

    def get_train_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns data for training

        Parameters:
            none

        Parameters:
            none

        Returns:
            tuple (labels, data) : contains train dataset
        """
        return (self.train_labels, self.train_data)
    
    def get_train_data_generator(self) -> ImageDataGenerator:
        """
        Returns ImageDataGenerator to distort train data for better training

        Parameters:
            none

        Returns:
            ImageDataGenerator : distortions to perform on training data
        """
        return self.train_datagen