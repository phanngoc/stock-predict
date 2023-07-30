
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
        
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


class TrainRnn:
    def __init__(self):
        pass

    def fit(self, trainset):    
        self.scaler = MinMaxScaler(feature_range = (0,1))
        training_scaled = self.scaler.fit_transform(trainset)
        self.x_train = []
        self.y_train = []

        for i in range(10, len(training_scaled)):
            self.x_train.append(training_scaled[i-10:i, 0])
            self.y_train.append(training_scaled[i,0])

        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)

        self.train()

    def train(self):
        inputShape = (self.x_train.shape[1],1)
        self.regressor = Sequential()
        self.regressor.add(LSTM(units = 50,return_sequences = True, input_shape = inputShape))
        self.regressor.add(Dropout(0.2))
        self.regressor.add(LSTM(units = 50,return_sequences = True))
        self.regressor.add(Dropout(0.2))
        self.regressor.add(LSTM(units = 50,return_sequences = True))
        self.regressor.add(Dropout(0.2))
        self.regressor.add(LSTM(units = 50))
        self.regressor.add(Dropout(0.2))

        self.regressor.add(Dense(units = 1))

        self.regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')
        self.regressor.summary()
        self.regressor.fit(self.x_train, self.y_train, epochs = 50, batch_size = 32)
    
    def tranform_to_predict(self, data):
        new_data = []
        for i in range(0, len(data)):
            item = np.array(data[i]).reshape(-1,1)
            item = self.scaler.transform(item)
            new_data.append(item)
        new_data = np.array(new_data)
        new_data = np.reshape(new_data, (new_data.shape[0], new_data.shape[1], 1))

        return new_data

    def predict(self, data):
        raw_transform_data = self.tranform_to_predict(data)
        result = self.regressor.predict(raw_transform_data)
        predicted_price = self.scaler.inverse_transform(result)
        return predicted_price
