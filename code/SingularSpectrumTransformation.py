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

def main():
    df = get_para('N225', 20080901, 20090601)
    df['steady_alpha'] = make_steady(df['alpha'], 1)
    df['steady_alpha'] = preprocessing.minmax_scale(df['steady_alpha'])
    df['steady_beta'] = make_steady(df['beta'], 1)
    df['steady_beta'] = preprocessing.minmax_scale(df['steady_beta'])
    df['steady_gamma'] = make_steady(df['gamma'], 1)
    df['steady_gamma'] = preprocessing.minmax_scale(df['steady_gamma'])
    df['steady_delta'] = make_steady(df['delta'], 1)
    df['steady_delta'] = preprocessing.minmax_scale(df['steady_delta'])

    ratio = 0.8
    date = 20090101
    train_data = prepare_test_data_by_date(df, date)[0]
    test_data = prepare_test_data_by_date(df, date)[1]

    # train_vector = np.array([train_data['steady_alpha']]).T
    # test_vector = np.array([test_data['steady_alpha']]).T

    width = 5
    # 窓幅を使ってベクトルの集合を作成
    train_vector = embed(train_data['steady_alpha'], width)
    test_vector = embed(test_data['steady_alpha'], width)

    train_date = train_data.iloc[width-1:].index
    test_date = test_data.iloc[width-1:].index
    price_date = test_data.index

    w = 50 # width
    m = 2
    k = w/2
    L = k/2 # lag
    Tt = test_vector.size
    score = np.zeros(Tt)

    for t in range(w+k, Tt-L+1+1):
        tstart = t-w-k+1
        tend = t-1
        X1 = embed(test_data[tstart:tend], w).T[::-1, :] # trajectory matrix
        X2 = embed(test_data[(tstart+L):(tend+L)], w).T[::-1, :] # test matrix

    U1, s1, V1 = np.linalg.svd(X1, full_matrices=True)
    U1 = U1[:,0:m]
    U2, s2, V2 = np.linalg.svd(X2, full_matrices=True)
    U2 = U2[:,0:m]

    U, s, V = np.linalg.svd(U1.T.dot(U2), full_matrices=True)
    sig1 = s[0]
    score[t] = 1 - np.square(sig1)

    # 変化度をmax1にするデータ整形
    mx = np.max(score)
    score = score / mx

    # プロット
    test_for_plot = data[3001:6000, 2]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    p1, = ax1.plot(score, '-b')
    ax1.set_ylabel('degree of change')
    ax1.set_ylim(0, 1.2)
    p2, = ax2.plot(test_for_plot, '-g')
    ax2.set_ylabel('original')
    ax2.set_ylim(0, 12.0)
    plt.title("Singular Spectrum Transformation")
    ax1.legend([p1, p2], ["degree of change", "original"])
    plt.savefig('./results/sst.png')
    plt.show()

def embed(lst, dim):
    emb = np.empty((0,dim), float)
    for i in range(lst.size - dim + 1):
    tmp = np.array(lst[i:i+dim]).reshape((1,-1))
    emb = np.append( emb, tmp, axis=0)
    return emb

if __name__ == '__main__':
    main()