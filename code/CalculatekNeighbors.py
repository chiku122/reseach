import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from GetPara import get_para
from sklearn import preprocessing
from MakeSteady import make_steady
from PrepareTestData import prepare_test_data_by_date
from statsmodels.tsa.stattools import adfuller


def kneighbors(stock_index, train_first_date, train_last_date, test_first_date, test_last_date, k, width):
    df = get_para(stock_index, train_first_date, test_last_date)
    df['alpha'] = make_steady(df['alpha'], 1)
    df['alpha'] = preprocessing.minmax_scale(df['alpha'])

    train_data = prepare_test_data_by_date(df, train_last_date, test_first_date)[0]
    test_data = prepare_test_data_by_date(df, train_last_date, test_first_date)[1]

    print('ksvalue:', adfuller(train_data['alpha'])[1])

    # 窓幅を使ってベクトルの集合を作成
    train_vector = embed(train_data['alpha'], width)
    test_vector = embed(test_data['alpha'], width)

    # k近傍法でクラスタリング
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(train_vector)

    # 距離を計算
    d = neigh.kneighbors(test_vector)[0]
    d_k = d[:, k-1]
    d_max = max(d_k)

    # 距離をtest_dataの列に追加
    d_k = np.append(np.array([np.nan]*(width-1)), d_k)
    test_data.loc[:, 'error'] = d_k

    data = test_data['error'].dropna().values
    print(max(data))
            

    # グラフ作成
    fig, ax = plt.subplots(1,1, tight_layout=True)

    # 異常度
    error_ax = ax
    error_ax.plot(test_data.index, test_data['error'], color='red')
    error_ax.set_xlabel("date")
    error_ax.set_ylabel("error")
    error_ax.legend()
    error_ax.grid()
    # error_ax.axvline(test_data[test_data['error'] == d_max].index, linestyle='--')
    error_ax.vlines(test_data[test_data['error'] >= 1.3].index, ymin=0, ymax=1.0, alpha=0.2, color='red')
    error_ax.tick_params(axis='x', rotation=30)

    price_ax = ax.twinx()
    price_ax.plot(test_data.index, test_data['price'], color='black', label='stock prices', linestyle='solid', alpha=1.0, linewidth=0.5)
    # price_ax.plot(test_data.index, test_data['price'].rolling(window=50, center=True).mean(), color='yellowgreen', label='price')
    price_ax.set_ylabel('stock prices')

    plt.show()

def embed(lst, dim):
    emb = np.empty((0, dim), float)
    for i in range(lst.size - dim + 1):
        tmp = np.array(lst[i:i+dim])[::-1].reshape((1, -1))
        emb = np.append(emb, tmp, axis=0)
    return emb
    
kneighbors('N225', 20040101, 20080101, 20080101, 20240101, 5, 5)