import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from GetPara import get_para
from sklearn import preprocessing
from MakeSteady import make_steady
from SeasonalDecompose import plot_stl
from PrepareTestData import prepare_test_data_by_date
import datetime as dt
from statsmodels.tsa.stattools import adfuller


def kneighbors(stock_index, train_first_date, train_last_date, test_first_date, test_last_date, k, width):
    df = get_para(stock_index, train_first_date, test_last_date)
    df['ksvalue'] = make_steady(df['ksvalue'], 0)
    # df['ksvalue'] = preprocessing.minmax_scale(df['ksvalue'])

    train_data = prepare_test_data_by_date(df, train_last_date, test_first_date)[0]
    test_data = prepare_test_data_by_date(df, train_last_date, test_first_date)[1]

    print('KS value:', adfuller(train_data['ksvalue'])[1])

    # train_vector = [[0,0,0,0]]*len(train_data)
    # for i in range(len(train_data)):
    #     train_vector[i] = [train_data['alpha'][i], train_data['beta'][i], train_data['gamma'][i], train_data['delta'][i]]
    # test_vector = [[0,0,0,0]]*len(test_data)
    # for i in range(len(test_data)):
    #     test_vector[i] = [test_data['alpha'][i], test_data['beta'][i], test_data['gamma'][i], test_data['delta'][i]]

    # 窓幅を使ってベクトルの集合を作成
    train_vector = embed(train_data['ksvalue'], width)
    test_vector = embed(test_data['ksvalue'], width)

    # k近傍法でクラスタリング
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(train_vector)

    # 距離を計算
    d = neigh.kneighbors(test_vector)[0]
    d_k = d[:, k-1]
    d_max = max(d_k)

    # 距離の正規化
    # d_k = preprocessing.minmax_scale(d_k)

    # 距離をtest_dataの列に追加
    d_k = np.append(np.array([np.nan]*(width-1)), d_k)
    test_data.loc[:, 'error'] = d_k

    # グラフ作成
    fig, ax = plt.subplots(1,1, tight_layout=True)

    # 異常度
    error_ax = ax
    error_ax.plot(test_data.index, test_data['error'], color='orange', label='KS value')
    error_ax.set_xlabel("date")
    error_ax.set_ylabel("error")
    error_ax.legend()
    error_ax.grid()
    error_ax.axvline(test_data[test_data['error'] == d_max].index, linestyle='--')
    error_ax.vlines(test_data[test_data['error'] >= 0.8].index, ymin=0, ymax=1.0)
    error_ax.tick_params(axis='x', rotation=30)

    price_ax = ax.twinx()
    price_ax.plot(test_data.index, test_data['price'], color='green', label='stock prices', linestyle='dashed', alpha=0.4)
    # price_ax.plot(test_data.index, test_data['price'].rolling(window=50, center=True).mean(), color='yellowgreen', label='price')
    price_ax.set_ylabel('stock prices')

    plt.show()

def embed(lst, dim):
    emb = np.empty((0, dim), float)
    for i in range(lst.size - dim + 1):
        tmp = np.array(lst[i:i+dim])[::-1].reshape((1, -1))
        emb = np.append(emb, tmp, axis=0)
    return emb
    
kneighbors('N225', 20050101, 20080101, 20080101, 20100101, 5, 5)