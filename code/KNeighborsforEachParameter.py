import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from GetPara import get_para
from sklearn import preprocessing
from MakeSteady import make_steady
from PrepareTestData import prepare_test_data_by_date
from statsmodels.tsa.stattools import adfuller
from BryBoschan import bry_boschan


def each_kneighbors(stock_index, train_first_date, train_last_date, test_first_date, test_last_date, k, width):
    df = get_para(stock_index, train_first_date, test_last_date)
    df['alpha'] = make_steady(df['alpha'], 1)
    df['alpha'] = preprocessing.minmax_scale(df['alpha'])
    df['beta'] = make_steady(df['beta'], 1)
    df['beta'] = preprocessing.minmax_scale(df['beta'])
    df['gamma'] = make_steady(df['gamma'], 1)
    df['gamma'] = preprocessing.minmax_scale(df['gamma'])
    df['delta'] = make_steady(df['delta'], 1)
    df['delta'] = preprocessing.minmax_scale(df['delta'])

    train_data = prepare_test_data_by_date(df, train_last_date, test_first_date)[0]
    test_data = prepare_test_data_by_date(df, train_last_date, test_first_date)[1]

    print('alpha:', adfuller(train_data['alpha'])[1])
    print('beta:', adfuller(train_data['beta'])[1])
    print('gamma:', adfuller(train_data['gamma'])[1])
    print('delta:', adfuller(train_data['delta'])[1])

    # 窓幅を使ってベクトルの集合を作成
    train_vector_alpha = embed(train_data['alpha'], width)
    test_vector_alpha = embed(test_data['alpha'], width)
    train_vector_beta = embed(train_data['beta'], width)
    test_vector_beta = embed(test_data['beta'], width)
    train_vector_gamma = embed(train_data['gamma'], width)
    test_vector_gamma = embed(test_data['gamma'], width)
    train_vector_delta = embed(train_data['delta'], width)
    test_vector_delta = embed(test_data['delta'], width)

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(train_vector_alpha)
    d_alpha = neigh.kneighbors(test_vector_alpha)[0]
    d_k_alpha = d_alpha[:, k-1]
    d_k_alpha = np.append(np.array([np.nan]*(width-1)), d_k_alpha)
    test_data.loc[:, 'alpha_error'] = d_k_alpha

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(train_vector_beta)
    d_beta = neigh.kneighbors(test_vector_beta)[0]
    d_k_beta = d_beta[:, k-1]
    d_k_beta = np.append(np.array([np.nan]*(width-1)), d_k_beta)
    test_data.loc[:, 'beta_error'] = d_k_beta
    
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(train_vector_gamma)
    d_gamma = neigh.kneighbors(test_vector_gamma)[0]
    d_k_gamma = d_gamma[:, k-1]
    d_k_gamma = np.append(np.array([np.nan]*(width-1)), d_k_gamma)
    test_data.loc[:, 'gamma_error'] = d_k_gamma

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(train_vector_delta)
    d_delta = neigh.kneighbors(test_vector_delta)[0]
    d_k_delta = d_delta[:, k-1]
    d_k_delta = np.append(np.array([np.nan]*(width-1)), d_k_delta)
    test_data.loc[:, 'delta_error'] = d_k_delta

    # グラフ作成
    fig, ax = plt.subplots(1,1, tight_layout=True)

    # 異常度
    error_ax = ax
    # error_ax.plot(test_data.index, test_data['error'], color='orange', label='KS value')
    error_ax.set_xlabel("date")
    error_ax.set_ylabel("error")
    error_ax.legend()
    error_ax.grid()
    # ln1 = error_ax.plot(test_data['alpha_error'], color='r', label='alpha')
    # ln2 = error_ax.plot(test_data['beta_error'], color='b', label='beta')
    # ln3 = error_ax.plot(test_data['gamma_error'], color='y', label='gamma')
    # ln4 = error_ax.plot(test_data['delta_error'], color='g', label='delta')
    # error_ax.axvline(test_data[test_data['error'] == d_max].index, linestyle='--')
    # error_ax.vlines(test_data[test_data['alpha_error'] >= 0.4].index, ymin=0, ymax=1.0, alpha=0.2, color='red')
    # error_ax.vlines(test_data[test_data['beta_error'] >= 0.9].index, ymin=0, ymax=1.0, alpha=0.2, color='blue')
    # error_ax.vlines(test_data[test_data['gamma_error'] >= 0.9].index, ymin=0, ymax=1.0, alpha=0.2, color='yellow')
    # error_ax.vlines(test_data[test_data['delta_error'] >= 0.9].index, ymin=0, ymax=1.0, alpha=0.2, color='green')
    # for index in test_data.index:
    #     if test_data['alpha_error'][index] >= 0.5 and test_data['beta_error'][index] >= 0.5 and test_data['delta_error'][index] >= 0.5:
    #         error_ax.vlines(index, ymin=0, ymax=1.0, alpha=0.2, color='purple')
    error_ax.tick_params(axis='x', rotation=30)
    # error_ax.set_ylim(0, 2.5)

    price_ax = ax.twinx()
    price_ax.plot(test_data.index, test_data['price'], color='black', label='stock prices', linestyle='solid', alpha=1.0, linewidth=0.5)
    # price_ax.plot(test_data.index, test_data['price'].rolling(window=50, center=True).mean(), color='yellowgreen', label='price')
    price_ax.set_ylabel('stock prices')
    h1, l1 = error_ax.get_legend_handles_labels()
    h2, l2 = price_ax.get_legend_handles_labels()
    error_ax.legend(h1+h2, l1+l2)

    # Bry-Boschan法を適用
    peaks, troughs = bry_boschan(df['price'], 10,30)


    plt.title('N225')
    plt.show()

def embed(lst, dim):
    emb = np.empty((0, dim), float)
    for i in range(lst.size - dim + 1):
        tmp = np.array(lst[i:i+dim])[::-1].reshape((1, -1))
        emb = np.append(emb, tmp, axis=0)
    return emb

each_kneighbors('N225', 20040101, 20080101, 20080101, 20240101, 5, 5)