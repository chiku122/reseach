import matplotlib.pyplot as plt
from sklearn import preprocessing
from GetPara import get_para
import statsmodels.api as sm
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import datetime as dt

def prepare_test_data_by_ratio(time_series, ratio):
    train_size = int(len(time_series) * ratio)
    test_size = len(time_series) - train_size

    train_data = time_series[:train_size]
    test_data = time_series[train_size:]
    
    return(
        [train_data, test_data]
    )

def prepare_test_data_by_date(df, train_last_date, test_first_date):
    train_last_date_str = dt.datetime.strptime(str(train_last_date), '%Y%m%d').strftime('%Y-%m-%d')
    test_first_date_str = dt.datetime.strptime(str(test_first_date), '%Y%m%d').strftime('%Y-%m-%d')
    train_data = df[:train_last_date_str]
    test_data = df[test_first_date_str:]
    
    return(
        [train_data, test_data]
    )