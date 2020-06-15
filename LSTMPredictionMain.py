from data_prep import build_comparison_data, build_train_data
from sklearn.preprocessing import MinMaxScaler
from predefines import obtain_data_from_csv
from train import stock_prediction_LSTM
from train_preprocessed_data import stock_prediction__preprocesed_data_LSTM
from plot import plot_data
import fileinput
import pickle

menu_action = [ '1. Download data',
                '2. Plot data',
                '3. Train new LSTM model and predict',
                '4. Predict using existing LSTM model',
                '5. Train new LSTM model trained on preprocessed data and predict',
                '6. Predict using existing LSTM model (data of Eray)',
                '7. Exit' 
                ]

if __name__ == '__main__':

    filename = 'BABA'

    preprocessed = ['AMD_data']

    while True:
        
        print('Options:')
        for item in menu_action:
            print(item)

        action = None

        for line in fileinput.input():
            if line.rstrip() not in "1234567" or len(line.rstrip()) != 1:
                print('Invalid input, try again!!!')
            else:
                action = line.rstrip()
                fileinput.close()

        if action == '1':
            
            build_train_data()
            build_comparison_data()

        elif action == '2':

            if filename in preprocessed:
                print('The data chosen is preprocessed. Choose options 5 or 6 one.')
            else:
                plot_data(filename)

        elif action == '3':

            if filename in preprocessed:
                print('The data chosen is preprocessed. Choose options 5 or 6 one.')
            else:
                stock_prediction_LSTM(filename,True)

        elif action == '4':

            if filename in preprocessed:
                print('The data chosen is preprocessed. Choose options 5 or 6 one.')
            else:
                stock_prediction_LSTM(filename,False)
        
        elif action == '5':

            if filename in preprocessed:
                stock_prediction__preprocesed_data_LSTM(filename,True)
            else:
                print('The data chosen is not preprocessed. Try another one.')
        
        elif action == '6':

            if filename in preprocessed:
                stock_prediction__preprocesed_data_LSTM(filename,False)
            else:
                print('The data chosen is not preprocessed. Try another one.')

        elif action == '7':

            break

