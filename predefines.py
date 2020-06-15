from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import pandas as pd
import os



EPOCHS = 1000
BATCH_SIZE = 32
TIME_STEPS = 60
PREDICT_PERIOD = 5

#2 layers 100 and 50


def create_model(data):
    regressor = Sequential()
    regressor.add(LSTM(units = 64, return_sequences = True, input_shape = (data.shape[1], 1),kernel_initializer='random_uniform'))
    regressor.add(LSTM(units = 64, return_sequences = False))
    regressor.add(Dropout(0.3))
    regressor.add(Dense(1))
    regressor.compile(optimizer ='adam' , loss = 'mean_squared_error')
    return regressor

def obtain_data_from_csv(path):
    df_train = pd.read_csv(os.path.join(os.path.dirname(__file__), '.', path+'.csv'))

    data = pd.DataFrame(columns = ['Date', 'Open', 'High', 'Low', 'Close','Adj Close'])

    data['Date'] = df_train.loc[:,'Date']
    data['Open'] = df_train.loc[:,'Open']
    data['High'] = df_train.loc[:,'High']
    data['Low'] = df_train.loc[:,'Low']
    data['Close'] = df_train.loc[:,'Close']
    data['Adj Close'] = df_train.loc[:,'Adj Close']

    return data
