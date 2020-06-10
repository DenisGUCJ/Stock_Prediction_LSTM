from data_prep import build_comparison_data, build_train_data
from sklearn.preprocessing import MinMaxScaler
from predefines import obtain_data_from_csv
from train import stock_prediction_LSTM
from plot import plot_data
import fileinput
import pickle

menu_action = [ '1. Download data',
                '2. Plot data',
                '3. Train new LSTM model and predict',
                '4. Predict using existing LSTM model',
                '5. Exit' 
                ]

if __name__ == '__main__':

    filename = 'AAPL'

    while True:
        
        print('Options:')
        for item in menu_action:
            print(item)

        action = None

        for line in fileinput.input():
            if line.rstrip() not in "12345" or len(line.rstrip()) != 1:
                print('Invalid input, try again!!!')
            else:
                action = line.rstrip()
                fileinput.close()

        if action == '1':
            
            build_train_data()
            build_comparison_data()

        elif action == '2':

            plot_data(filename)

        elif action == '3':

            stock_prediction_LSTM(filename,True)

        elif action == '4':

            stock_prediction_LSTM(filename,False)

        elif action == '5':

            break

3