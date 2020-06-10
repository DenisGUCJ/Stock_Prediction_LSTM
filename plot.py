import os
import csv
from datetime import date, datetime
import matplotlib.pyplot as plt
from predefines import obtain_data_from_csv

def plot_data(filename):

    dates = []
    data = obtain_data_from_csv('data/' + filename)
    for item in data['Date']:
        dates.append(datetime.strptime(item, '%Y-%m-%d').date())
    
    plt.figure(figsize=(20,10))
    plt.plot(dates, data['Close'], color='red', label='Price model')  
    plt.plot(dates, data['Open'], color='green', label='Price model')  
    plt.xlabel('Date') 
    plt.ylabel('Adj Close')
    plt.title('Price History')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    filename = 'AAPL'
    plot_data(filename)