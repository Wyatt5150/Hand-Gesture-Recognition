'''
Ymai's notes
    added proper type hinting

    Renamed SignLanguageMNISTDataModule to DataModule
        this class isnt specialized just for the signlanguageMNIST data set
        there is no reason that you needed to make the class name that long

    CustomDataset
        added prepare_data()
            fixes the way mnist labeled the data

    DataModule
        init
            removed parameters: train_csv, val_csv, test_csv
                removed parameters replaced with one parameter called: dataset
                datasets should be named '<dataset>_train.csv' and '<dataset>_test.csv'
                datasets should be place in their respective folders in data
        setup()
            parameter:stage is never used so i am removing it
            fixed directory paths
        removed prepare_data()
            it fit more naturally into CustomDataset so I moved it
'''

import os  # Import os for file path management
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

class DataModule():
    def __init__(self, dataset:str):
        """
        Initializes the DataModule based on specified path

        Args:
            dataset (str): name of dataset being used
        """
        self.train_csv = dataset+'_train.csv'
        self.val_csv = dataset+'_train.csv'
        self.test_csv = dataset+'_test.csv'

        # load data sets
        curDir = os.getcwd()
        train_df=pd.read_csv(os.path.join(curDir, 'data','train', dataset+'_train.csv'))
        test_df=pd.read_csv(os.path.join(curDir, 'data','test', dataset+'_test.csv'))

        # separate labels from the rest of the data
        train_labels = train_df['label']
        self.train_data = train_df.drop(['label'],axis=1)
        self.train_data = self.train_data.values.reshape(-1,28,28,1)

        self.test_labels = test_df['label']
        self.test_data = test_df.drop(['label'],axis=1)
        self.test_data = self.test_data.values.reshape(-1,28,28,1)
        self.test_data = self.test_data/255

        # converts labels into binary representation
        #lb=LabelBinarizer()
        #train_labels=lb.fit_transform(train_labels)
        #test_labels=lb.fit_transform(test_labels)
        print(train_labels)
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

        Returns:
            returns tuple (labels, data) for test dataset
        """
        return tuple(self.test_labels, self.test_data)

    def get_train_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns data for training

        Returns:
            returns tuple (labels, data) for train dataset
        """
        return tuple(self.train_labels, self.train_data)
    
    def get_train_data_generator(self):
        return self.train_datagen