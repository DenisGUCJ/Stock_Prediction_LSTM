from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
import time
import fileinput
import pickle
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from predefines import create_model
from predefines import BATCH_SIZE, PREDICT_PERIOD, TIME_STEPS, EPOCHS
from matplotlib import pyplot as plt
from predefines import obtain_data_from_csv
from collections import deque




def stock_prediction_LSTM(filename, upadteFlag):


    data = obtain_data_from_csv('data/'+filename)
    size = data['Open'].size

    #preprocess the data

    train_set = data.iloc[:,1:2].values #Open prices
    temp_set = data.iloc[:,4:5].values #Closed prices

    scaler = MinMaxScaler(feature_range=(0,1))
    close_scaler = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = scaler.fit_transform(train_set)
    temp_set_scaled = close_scaler.fit_transform(temp_set)
    X_train = []
    y_train = []


    #creating timeseries format of data

    for i in range(TIME_STEPS, size):
        X_train.append(training_set_scaled[i-TIME_STEPS:i, 0])
        #y_train.append(training_set_scaled[i, 0])
        y_train.append(temp_set_scaled[i, 0])  

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



    #testing


    testdata = obtain_data_from_csv('dataComparison/'+filename)

    size2 = testdata['Close'].size

    real_stock_price = testdata.iloc[:, 4:5].values
    dataset_total = pd.concat((data['Open'], testdata['Open']), axis = 0)
    test_data = close_scaler.transform(real_stock_price)

    inputs = dataset_total[len(dataset_total) - len(testdata) - TIME_STEPS:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)
    
    
    X_test = []
    Y_test = []
    last_sequence = []

    for i in range(TIME_STEPS, size2 + TIME_STEPS - PREDICT_PERIOD):
        X_test.append(inputs[i-TIME_STEPS:i, 0])
        Y_test.append(test_data[i-TIME_STEPS,0])
        if i == size2 + TIME_STEPS - PREDICT_PERIOD - 1:
            last_sequence.append(inputs[i-TIME_STEPS:i, 0])



    X_test,Y_test,last_sequence = np.array(X_test), np.array(Y_test), np.array(last_sequence)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #last_sequenceest = np.reshape(last_sequence, (last_sequence.shape[0], last_sequence.shape[1], 1))


    mcp = ModelCheckpoint(os.path.join(os.path.dirname(__file__), '.', 'models/lstm_model'), monitor='val_loss', verbose=1,
                          save_best_only=True, save_weights_only=False, mode='min', period=1)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                   patience=100, min_delta=0.0001)

    csv_logger = CSVLogger(os.path.join(os.path.dirname(__file__), '.','logs/training_log_' + time.ctime().replace(" ","_") + '.log'), append=True)

    model = None

    if upadteFlag == True:

        print('Creating new model...')
        model = create_model(X_train)
        history = model.fit(X_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose=1, validation_data=(X_test,Y_test), callbacks=[ mcp,csv_logger,es ])
        print('Saving model...')
        print(history.history)
        pickle.dump(model, open("models/lstm_model","wb"))
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
            model = pickle.load(open("models/lstm_model", 'rb'))
            print("Loaded saved model...")
        except FileNotFoundError:
            print('Model not found')

    #prediction


    predicted_stock_price = model.predict(X_test)
    test_stock_prices = close_scaler.inverse_transform(predicted_stock_price)

    #prediction
    prediction = []

    for i in range(0,PREDICT_PERIOD):
        last_element = predicted_stock_price[-1]
        #last_sequence = np.delete(np.append(last_sequence,last_element),0)
        #last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], 1))
        #predicted_elem = model.predict(last_sequence)
        #prediction.append(predicted_elem)
        #predicted_stock_price = np.append(predicted_stock_price, predicted_elem)
        last_sequence = np.delete(np.append(last_sequence,last_element),0)
        last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], 1))
        #predicted_elem = model.predict(last_sequence)
        #prediction.append(predicted_elem)
        #predicted_stock_price = np.append(predicted_stock_price, predicted_elem)
        #X_test = np.append(X_test,last_sequence)
        X_test = np.vstack([X_test,last_sequence])
        #X_test = np.array(X_test)
        #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = model.predict(X_test)

    
    
    prediction = close_scaler.inverse_transform(predicted_stock_price)

    

    x_pred = list(range(prediction.shape[0] - PREDICT_PERIOD - 1,prediction.shape[0]))

    print('Ploting the prediction...')
    plt.figure(figsize=(20,10))
    plt.plot(real_stock_price, color = 'green', label = 'Stock Price')
    plt.plot(prediction, color = 'blue',marker='o', label = 'Predicted Stock Price')
    plt.plot(test_stock_prices, color = 'red', label = 'Test Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Trading Day')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
