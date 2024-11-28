import matplotlib.pyplot as plt
import numpy as np
from itertools import islice
from GetPara import get_para

# SSA 用の関数
def window(seq, n):
    """
    window 関数で要素を1づつずらした2次元配列を出す. 戻り値は generator
    """
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
def SSA_anom(test, traject, w, ncol_t, ncol_h, ns_t, ns_h,
             normalize=False):
    """
    特異スペクトル分析 (SSA) による時系列の特徴づけ
    ARGUMENTS:
    -------------------------------------------------
    test: array-like. テスト行列を作る部分時系列
    tracject: array-like. 履歴行列を作る部分時系列
    ns_h: 履歴行列から取り出す特異ベクトルの数
    ns_t: テスト行列から取り出す特異ベクトルの数
    -------------------------------------------------
    RETURNS:
    3要素のタプル: 
        要素1: 2つの部分時系列を比較して求めた異常度
        要素2, 3: テスト行列・履歴行列をそれぞれの特異値の累積寄与率 
    """
    H_test = np.array(
        tuple(x[:ncol_t] for x in window(test, w))[:w]
    )  # test matrix
    H_hist = np.array(
        tuple(x[:ncol_h] for x in window(traject, w))[:w]
    )  # trajectory matrix
    if normalize:
        H_test = (H_test - H_test.mean(axis=0,
                                       keepdims=True)) / H_test.std(axis=0)
        H_hist = (H_hist - H_hist.mean(axis=0,
                                       keepdims=True)) / H_hist.std(axis=0)
    Q, s1 = np.linalg.svd(H_test)[0:2]
    Q = Q[:, 0:ns_t]
    ratio_t = sum(s1[0:ns_t]) / sum(s1)
    U, s2 = np.linalg.svd(H_hist)[0:2]
    U = U[:, 0:ns_h]
    ratio_h = sum(s2[0:ns_t]) /sum(s2)
    anom = 1 - np.linalg.svd(np.matmul(U.T, Q),
                                     compute_uv=False
                                    )[0]
    return (anom, ratio_t, ratio_h)

def SSA_CD(series, w, lag,
           ncol_h=None, ncol_t=None,
           ns_h=None, ns_t=None,
           standardize=False, normalize=False, fill=True):
    """
    Change Detection by Singular Spectrum Analysis
    SSA を使った変化点検知
    -------------------------------------------------
    w   : window width (= row width of matrices) 短いほうが感度高くなる
    lag : default=round(w / 4)  Lag among 2 matrices 長いほうが感度高くなる
    ncol_h: 履歴行列の列数 
    ncol_t: テスト行列の列数
    ns_h: 履歴行列から取り出す特異ベクトルの数. default=1 少ないほうが感度高くなる
    ns_t: テスト行列から取り出す特異ベクトルの数. default=1 少ないほうが感度高くなる
    standardize: 変換後の異常度の時系列を積分面積1で規格化するか
    fill: 戻り値の要素数を NaN 埋めで series と揃えるかどうか
    -------------------------------------------------
    Returns
    list: 3要素のリスト
        要素1: 2つの部分時系列を比較して求めた異常度のリスト
        要素2, 3: テスト行列・履歴行列をそれぞれの特異値の累積寄与率のリスト
    """
    if ncol_h is None:
        ncol_h = round(w / 2)
    if ncol_t is None:
        ncol_t = round(w / 2)
    if ns_h is None:
        ns_h = np.min([1, ncol_h])
    if ns_t is None:
        ns_t = np.min([1, ncol_t])
    if min(ncol_h, ncol_t) > w:
        print('ncol and ncol must be <= w')
    if ns_h > ncol_h or ns_t > ncol_t:
        print('I recommend to set ns_h >= ncol_h and ns_t >= ncol_t')
    start_at = lag + w + ncol_h
    end_at = len(series) + 1
    res = []
    for t in range(start_at, end_at):
        res = res + [SSA_anom(series[t - w - ncol_t:t],
                              series[t - lag - w - ncol_h:t - lag],
                              w=w, ncol_t=ncol_t, ncol_h=ncol_h,
                              ns_t=ns_t, ns_h=ns_h,
                              normalize=normalize)]
    anom = [round(x, 14) for x, r1, r2 in res]
    ratio_t = [r1 for x, r1, r2 in res]
    ratio_h = [r2 for x, r1, r2 in res]
    if fill == True:
        anom = [np.nan] * (start_at - 1) + anom
    if standardize:
        c = np.nansum(anom)
        if c != 0:
            anom = [x / c for x in anom]
    return [anom, ratio_t, ratio_h]

df = get_para('N225', 20080101, 20240101)

score = SSA_CD(series=df['alpha'], standardize=False, w=5, lag=5, ncol_h=3, ncol_t=3, ns_h=1, ns_t=1)
# score2 = SSA_CD(series=df['alpha'], standardize=False, w=20, lag=20, ns_h=1, ns_t=1)
# score3 = SSA_CD(series=df['alpha'], standardize=False, w=50, lag=20, ns_h=1, ns_t=1)


# グラフ作成
fig, ax = plt.subplots(1,1, tight_layout=True)

# 異常度
error_ax = ax
# error_ax.plot(df.index, score[0], color='red')
# error_ax.plot(df.index, score2[0], color='blue', label='w=20')
# error_ax.plot(df.index, score3[0], color='green', label='w=50')
error_ax.set_xlabel("date")
error_ax.set_ylabel("error")
error_ax.legend()
error_ax.grid()
# error_ax.axvline(test_data[test_data['error'] == d_max].index, linestyle='--')
for i in range(len(score[0])):
    if score[0][i] >= 0.0000010:
        error_ax.axvline(df.index[i], alpha=0.2, color='red')
error_ax.tick_params(axis='x', rotation=30)

price_ax = ax.twinx()
price_ax.plot(df.index, df['price'], color='black', label='stock prices', linestyle='solid', alpha=1.0, linewidth=0.5)
# price_ax.plot(test_data.index, test_data['price'].rolling(window=50, center=True).mean(), color='yellowgreen', label='price')
price_ax.set_ylabel('stock prices')

plt.show()
