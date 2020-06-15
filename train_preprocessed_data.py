from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
import time
import fileinput
import pickle
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from predefines import create_model
from predefines import BATCH_SIZE, PREDICT_PERIOD, TIME_STEPS, EPOCHS
from matplotlib import pyplot as plt
from predefines import obtain_data_from_csv
from collections import deque

TEST_PERIOD = 120


def stock_prediction__preprocesed_data_LSTM(filename, upadteFlag):

    df_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', filename+'.csv'))

    data = pd.DataFrame(columns = ['Open','Close'])

    data['Close'] = df_train.loc[:,'Close']
    data['Open'] = df_train.loc[:,'Open']

    

    #preprocess the data

    train_set = data.iloc[:,0:1].values #Open prices
    temp_set = data.iloc[:,1:2].values #Closed prices
    train_set = train_set[:train_set.size - TEST_PERIOD - PREDICT_PERIOD - TIME_STEPS]
    temp_set = temp_set[:temp_set.size - TEST_PERIOD - PREDICT_PERIOD - TIME_STEPS]

    size = train_set.size

    X_train = []
    y_train = []


    #creating timeseries format of data

    for i in range(TIME_STEPS, size):
        X_train.append(train_set[i-TIME_STEPS:i, 0])
        y_train.append(temp_set[i, 0])  

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



    #testing


    testdata = obtain_data_from_csv('dataComparison/'+filename)

    

    test_set = testdata.iloc[:,3:4].values 
    real_stock_price = testdata.iloc[:, 4:5].values

    scaler = MinMaxScaler(feature_range=(-1,1))
    close_scaler = MinMaxScaler(feature_range=(-1,1))

    test_set = scaler.fit_transform(test_set)
    real_stock_price =close_scaler.fit_transform(real_stock_price)
    
    test_set = test_set[test_set.shape[0]- TEST_PERIOD - TIME_STEPS:]
    real_stock_price = real_stock_price[real_stock_price.shape[0]- TEST_PERIOD:]
    size2 = real_stock_price.size

    #inputs = dataset_total[].values
    test_set = test_set.reshape(-1,1)
    #test_set = scaler.fit_transform(test_set)
    
    X_test = []
    Y_test = []
    last_sequence = []

    for i in range(TIME_STEPS, size2 + TIME_STEPS - PREDICT_PERIOD):
        X_test.append(test_set[i-TIME_STEPS:i, 0])
        Y_test.append(real_stock_price[i-TIME_STEPS,0])
        if i == size2 + TIME_STEPS - PREDICT_PERIOD - 1:
            last_sequence.append(test_set[i-TIME_STEPS:i, 0])



    X_test,Y_test,last_sequence = np.array(X_test), np.array(Y_test), np.array(last_sequence)
    X_test = np.reshape(X_test, (X_test.shape[0], TIME_STEPS, 1))


    mcp = ModelCheckpoint(os.path.join(os.path.dirname(__file__), '.', 'models/best_lstm_model_'+filename), monitor='val_loss', verbose=1,
                          save_best_only=True, save_weights_only=False, mode='min', period=1)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                   patience=100, min_delta=0.00001)

    csv_logger = CSVLogger(os.path.join(os.path.dirname(__file__), '.','logs/training_log_' + time.ctime().replace(" ","_") + '.log'), append=True)

    model = None

    if upadteFlag == True:

        print('Creating new model...')
        model = create_model(X_train)
        history = model.fit(X_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose=1, validation_data=(X_test,Y_test), callbacks=[ mcp,csv_logger,es ])
        print('Saving model...')
        print(history.history)
        pickle.dump(model, open("models/lstm_model_"+filename,"wb"))
        plt.figure(figsize=(20,10))
        plt.plot(history.history['loss'], color = 'blue', label = 'Train')
        plt.plot(history.history['val_loss'], color = 'green', label = 'Test')
        plt.title('Model loss')
        plt.xlabel(str(EPOCHS)+' Epochs')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.legend()
        plt.show()
    else:
        try:
            model = load_model(os.path.join(os.path.dirname(__file__),'models/',"best_lstm_model_"+filename))
            print("Loaded saved model...")
        except FileNotFoundError:
            print('Model not found')

        #prediction


        predicted_stock_price = model.predict(X_test)

        test_prediction = predicted_stock_price

        test_prediction = close_scaler.inverse_transform(test_prediction)

        #prediction
        prediction = []

        for i in range(0,PREDICT_PERIOD):
            last_element = predicted_stock_price[-1]
            last_element = last_element.reshape(1,1)
            last_sequence = np.delete(np.append(last_sequence,last_element),0)
            last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], 1))
            X_test = np.vstack([X_test,last_sequence])
            predicted_stock_price = model.predict(X_test)

        
        
        prediction = predicted_stock_price
        prediction = prediction[prediction.size - PREDICT_PERIOD:]
        prediction = close_scaler.inverse_transform(prediction)

        real_stock_price = close_scaler.inverse_transform(real_stock_price)

        #errors
        
        for i in range(0,PREDICT_PERIOD):
            error = ((real_stock_price[-PREDICT_PERIOD + i]-prediction[i])/real_stock_price[-PREDICT_PERIOD + i]) * 100
            print("error_"+str(i+1)+" : "+str(error))

        print('Ploting the prediction...')
        plt.figure(figsize=(20,10))
        plt.plot(real_stock_price, color = 'green', label = 'Stock Price')
        plt.plot(range(real_stock_price.size - PREDICT_PERIOD,real_stock_price.size),prediction,color = 'blue',marker='o', label = 'Predicted Stock Price')
        plt.plot(test_prediction, color = 'red', label = 'Test Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Trading Day')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()
