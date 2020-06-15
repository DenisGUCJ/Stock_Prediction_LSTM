import os
import yfinance as yf
from pandas_datareader import data as pdr
from pathlib import Path

yf.pdr_override()

START_DATE_TRAIN = "2014-08-01"
END_DATE_TRAIN = "2019-07-06"
START_DATE_COMPARISION = "2019-07-07"
END_DATE_COMPARISON = "2019-12-24"

tickers = ['SPY', 'BABA', 'AAPL']


def build_train_data(start=START_DATE_TRAIN, end=END_DATE_TRAIN):
    Path("./data").mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        data = pdr.get_data_yahoo(ticker, start, end)
        data.to_csv("./data/" + ticker + ".csv")
    
def build_comparison_data(start=START_DATE_COMPARISION, end=END_DATE_COMPARISON):
    Path("./dataComparison").mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        data = pdr.get_data_yahoo(ticker, start, end)
        data.to_csv("./dataComparison/" + ticker + ".csv")

if __name__ == "__main__":
    build_train_data()
    build_comparison_data()