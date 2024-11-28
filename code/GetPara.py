import csv
import datetime as dt
import numpy as np
import pandas as pd

def get_para(stock_index, first_date, last_date):

    #過去20年間のパラメータを取得
    with open('./data/{}/param/20040101-20240601.csv'.format(stock_index), 'r') as f:
        reader = csv.reader(f)
        l = [row for row in reader]

    date = []
    price = []
    for i in range(len(l[0])):
        date.append(dt.datetime.strptime(l[0][i], '%Y-%m-%d %H:%M:%S'))
        price.append(float(l[1][i]))

    alpha = []
    beta = []
    gamma = []
    delta = []
    for i in range(len(l[2])):
        alpha.append(float(l[2][i]))
        beta.append(float(l[3][i]))
        gamma.append(float(l[4][i]))
        delta.append(float(l[5][i]))

    #指定した期間のパラメータを取得
    specified_date = []
    specified_price = []
    specified_alpha = []
    specified_beta = []
    specified_gamma = []
    specified_delta = []
    for i in range(len(date)):
        if((date[i] >= dt.datetime.strptime(str(first_date), '%Y%m%d')) & (date[i] < dt.datetime.strptime(str(last_date), '%Y%m%d'))):
            specified_date.append(date[i])
            specified_price.append(price[i])
            specified_alpha.append(alpha[i])
            specified_beta.append(beta[i])
            specified_gamma.append(gamma[i])
            specified_delta.append(delta[i])

    data = np.array([specified_date, specified_price, specified_alpha, specified_beta, specified_gamma, specified_delta]).T
    time_series = pd.DataFrame(data, columns=['date', 'price', 'alpha', 'beta', 'gamma', 'delta'])
    time_series['date'] = pd.to_datetime(time_series['date'])
    time_series = time_series.set_index('date')
    print('length of time series:', len(time_series))
    
    return(
        time_series
    )
