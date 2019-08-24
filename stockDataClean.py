import pandas as pd
import os
import re
import math
from scipy import stats

def get_raw_df(stock_folder):
    df = pd.DataFrame()
    stock_file_paths = [stock_folder+"/"+stock for stock in os.listdir("./" + stock_folder)]
    for stock_file in stock_file_paths:
        stock_csv = re.split("/", stock_file)[-1]
        ticker = re.split("\.", stock_csv)[0]

        file = open(stock_file, 'r')
        stock_df = pd.read_csv(file)
        file.close()

        stock_df = stock_df[['Date', 'Adj Close']]
        stock_df[ticker] = stock_df['Adj Close'].pct_change()
        stock_df.drop(columns = ['Adj Close'], inplace=True)
        
        if df.empty:
            df = stock_df
        
            continue 
        df = pd.merge(df, stock_df, how='outer', on='Date')
    return df

def get_start_date(raw_df):
    return raw_df['Date'][1]

def get_end_date(raw_df):
    return raw_df['Date'].iloc[-1]

def get_trading_days(raw_df):
    return len(raw_df)

def cleaned_df(raw_df):
    raw_df.drop(0, inplace=True)
    assert not raw_df.isnull().values.any(), "Inconsistent Dates or NaN values"
    return raw_df.drop("Date", axis = 1)

def get_df(stock_folder):
    return cleaned_df(get_raw_df(stock_folder))

def stock_mean_daily(df):
    df = df.values + 1
    adj_mean = stats.gmean(df, axis=0)
    return adj_mean - 1

def stock_mean_annualized(df):
    mean_daily = stock_mean_daily(df)
    return (((mean_daily + 1)**253) - 1)


def covariance_matrix_daily(df):
    return df.cov().values

def covariance_matrix_annualized(df):
    return covariance_matrix_daily(df)*253

def stock_std_daily(df):
    return df.std().values

def stock_std_annualized(df):
    return stock_std_daily(df)*math.sqrt(253)

def stocks(df):
    return df.columns.values


        
################################
#   Main
################################
stock_folder = "Sample Stock Data"
df = get_df(stock_folder)

stocks = stocks(df)
mu = stock_mean_annualized(df)
std = stock_std_annualized(df)
sigma = covariance_matrix_annualized(df)