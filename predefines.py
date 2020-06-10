from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import optimizers
import pandas as pd
import os



EPOCHS = 150
BATCH_SIZE = 20
TIME_STEPS = 60
PREDICT_PERIOD = 5


def create_model(data):
    regressor = Sequential()
    regressor.add(LSTM(units = 150, return_sequences = True, input_shape = (data.shape[1], 1)))
    regressor.add(Dropout(0.5))
    regressor.add(LSTM(units = 150, return_sequences = True))
    regressor.add(Dropout(0.5))
    regressor.add(LSTM(units = 150, return_sequences = False))
    regressor.add(Dropout(0.5))
    regressor.add(Dense(1, activation='linear'))
    optimizer = optimizers.Adam()
    regressor.compile(optimizer ='rmsprop' , loss = 'mean_squared_error')
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
