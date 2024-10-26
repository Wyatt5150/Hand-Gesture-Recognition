import os
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
from scripts.dataModule import DataModule

class SignLanguageCNN():
    def __init__(self) -> None:
        """
        Initializes the SignLanguageCNN model.
        """
        self.model=Sequential()
        self.model.add(Conv2D(128,kernel_size=(5,5),
                        strides=1,padding='same',activation='relu',input_shape=(28,28,1)))
        self.model.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))
        self.model.add(Conv2D(64,kernel_size=(2,2),
                        strides=1,activation='relu',padding='same'))
        self.model.add(MaxPool2D((2,2),2,padding='same'))
        self.model.add(Conv2D(32,kernel_size=(2,2),
                        strides=1,activation='relu',padding='same'))
        self.model.add(MaxPool2D((2,2),2,padding='same'))
                
        self.model.add(Flatten())

        self.model.add(Dense(units=512,activation='relu'))
        self.model.add(Dropout(rate=0.25))
        self.model.add(Dense(units=24,activation='softmax'))
        self.model.summary()

    def train(self, data:DataModule, batch_size:int=128, epochs:int=10, model_name:str='model') -> None:
        '''
        trains the model on given DataModule and saves model to model_name.keras

        Parameters:
            data: DataModule for the data the model will be trained with
            batch_size: number of samples used each pass
            epochs: number of epochs the training will run
            model_name: name of the file that the model will be saved to
            
        Returns:
            None
        '''
        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        test = data.get_test_data()
        train = data.get_train_data()
        self.model.fit(data.get_train_data_generator.flow(train[1],train[0],batch_size=200),
                epochs = 50,
                validation_data=(test[1],test[0]),
                shuffle=1
                )

        (ls,acc)=self.model.evaluate(x=test[1],y=test[0])
        print('MODEL ACCURACY = {}%'.format(acc*100))

        self.model.save(os.path.join(os.getcwd(),'models', model_name+'.keras'))
