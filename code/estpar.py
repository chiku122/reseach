# pythonによる安定分布の推定手法
# version 1.0 2020/9/20
# version 1.1 2020/9/23
# version 1.2 2020/10/24

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import pandas as pd
import statistics
import pandas.tseries.offsets as offsets
from matplotlib.patches import ArrowStyle
import time
#from tqdm import tqdm_notebook as tqdm
import levy
from scipy.stats import levy_stable
import sys
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
from datetime import datetime


#seaborn style
#seaborn settings
#sns.set(style="whitegrid", palette="muted", color_codes=True)
#sns.set_style("whitegrid", {'grid.linestyle': '--'})

# optional settings
#from scipy.stats import norm
#from scipy.stats import rankdata

# calculate weighted 特性関数
def phi(X, k):
    CF = 0.0
    for i in range(len(X)):
        CF = CF + np.exp(complex(0, k*X[i]))
    CF = CF/len(X)
    return CF

def estphi(k, alpha, beta, gamma, delta):
    if k >= 0:
        p = np.exp(complex(0,delta*k)-(gamma*k)**alpha*complex(1.0,-beta*np.tan(np.pi*alpha/2)))
    if k < 0:
        p = np.exp(complex(0,delta*k)-(-gamma*k)**alpha*complex(1.0,beta*np.tan(np.pi*alpha/2)))
    return p

# estimate stable parameters with weighted observations
def F_alphaW(X, k0, k1):
    cf0 = phi(X, k0)
    cf1 = phi(X, k1)
    alpha = (np.log(-(np.log(cf0).real))-np.log(-(np.log(cf1).real)))/(np.log(k0)-np.log(k1))
    if alpha > 2:
        alpha = 2.0
    return alpha

def F_gammaW(X, k0, k1):
    cf0 = phi(X, k0)
    cf1 = phi(X, k1)
    gamma = np.exp((np.log(k0)*np.log(-(np.log(cf1).real))-np.log(k1)*np.log(-(np.log(cf0).real)))/(np.log(-(np.log(cf0).real))-np.log(-(np.log(cf1).real))))
    assert gamma > 0, '期待するガンマは正, 出力値[{0}]'.format(gamma)
    return gamma

def F_betaW(X, k0, k1, alpha, gamma):
    cf0 = phi(X, k0)
    cf1 = phi(X, k1)
    if abs(alpha-1.0) < 0.01:
        beta = (np.pi/2)*((k1*np.log(cf0).imag-k0*np.log(cf1).imag)/(gamma*k0*k1*(np.log(k1)-np.log(k0))))
    else:
        beta = (k1*np.log(cf0).imag - k0*np.log(cf1).imag)/(gamma**alpha*np.tan(np.pi*alpha/2)*(k0**alpha*k1-k1**alpha*k0))
    if beta < -1:
        beta = -1
    if beta > 1:
        beta = 1
    return beta

def F_deltaW(X, k0, k1, alpha):
    cf0 = phi(X, k0)
    cf1 = phi(X, k1)
    if abs(alpha-1.0) < 0.01:
        delta = (k1*np.log(cf0).imag*np.log(k1)-k0*np.log(cf1).imag*np.log(k0))/(k0*k1*(np.log(k1)-np.log(k0)))
    else:
        delta = (k1**alpha*np.log(cf0).imag-k0**alpha*np.log(cf1).imag)/(k0*k1**alpha-k1*k0**alpha)
    return delta

# 最適なフーリエ空間点k1を見つける
def g(alpha, tau=2.5):
    del_alpha = 0.01
    # 関数定義
    f = lambda x: (alpha*x**(alpha-1)+tau)*np.exp(-(x**alpha+tau*x)) - ((alpha+del_alpha)*x**(alpha+del_alpha-1)+tau)*np.exp(-(x**(alpha+del_alpha)+tau*x))
    df = lambda x: (alpha*(alpha-1)*x**(alpha-2)-(alpha*x**(alpha-1)+tau)**2)*np.exp(-(x**alpha+tau*x)) + (((alpha+del_alpha)*x**(alpha+del_alpha-1)+tau)**2-(alpha+del_alpha)*(alpha+del_alpha-1)*x**(alpha+del_alpha-2))*np.exp(-(x**(alpha+del_alpha)+tau*x))
    
    # 初期値の設定
    x0 = 0.01*alpha**1.5*np.exp(alpha)
    
    # ニュートン法で解を計算
    while True:
        x = x0 - f(x0) / df(x0)
        if abs(x-x0) < 0.00001:
            break
        else:
            x0 = x
    if x > 1:
        assert x != 1, 'αとフーリエ空間点との関係性を表す関数gは1になってはいけない, 出力値[{0}]'.format(x)
        x = 0.2
    return x

# estimate stable parameters from (weighted) characteristic function

def est_stablepars(X, gamma0):
    #　入力: データ, データの分散（パラメータを推定するのにある程度妥当な初期スケールが必要だから）
    # Estimate the "temporary gamma"
    bound = 0.5
    k_begin, k_end = (1.0-bound)/gamma0, (1.0+bound)/gamma0 # この範囲で適切なフーリエ空間点k_0を探索する
    pars = np.zeros(4)
    
    k = np.arange(k_begin, k_end, (k_end-k_begin)/100)
    cf_real = []
    cf_imag = []
    
    for j in range(len(k)):
        cf = phi(X, k[j])
        cf_real = np.append(cf_real, cf.real)
        cf_imag = np.append(cf_imag, cf.imag)
        
    cf_abs = cf_real**2+cf_imag**2
    mse = (np.log(cf_abs)-(-1))**2
    gamma_tmp = 1.0/k[np.argmin(mse)]
    gamma_tmp # step 1
        
    # step 2
    k0_rough = 0.05/gamma_tmp # ここの分子の値は初期値なので1以下であれば何でも良い
    k1_rough = 1/gamma_tmp
        
    # step 3
    # rough estimate of alpha and gamma
    alpha_rough = F_alphaW(X, k0_rough, k1_rough)
    gamma_rough = F_gammaW(X, k0_rough, k1_rough)
        
    #compute rough estimate of moment point eta_tilde
    eta_rough = g(alpha_rough) # step 4
        
    # step 5
    # Recalculate the points
    k0_rough2 = eta_rough/gamma_rough
    k1_rough2 = 1.0/gamma_rough
        
    # step 6
    alpha_rough2 = F_alphaW(X, k0_rough2, k1_rough2)
    gamma_rough2 = F_gammaW(X, k0_rough2, k1_rough2)

    # step 7
    eta_rough2 = g(alpha_rough2)

    # step 8
    k0 = eta_rough2/gamma_rough2
    k1 = 1.0/gamma_rough2

    # step 9
    alpha = F_alphaW(X, k0, k1)
    gamma = F_gammaW(X, k0, k1)

    # step 10
    beta = F_betaW(X, k0, k1, alpha, gamma)
    delta = F_deltaW(X, k0, k1, alpha)
        
    pars = np.array([alpha, beta, gamma, delta])
    
    return pars

def estpar(X):
    return est_stablepars(X, np.var(X)*100) # gamma0 の値の決め方は任意性があるかもしれない
    
########################################
# 実行
# data は array型
# estpar(data)

my_share = share.Share('^N225')
symbol_data = None

try:
    symbol_data = my_share.get_historical(share.PERIOD_TYPE_YEAR,3,
                                          share.FREQUENCY_TYPE_DAY,1)
except YahooFinanceError as e:
    print(e.message)
    sys.exit(1)


symbol_data.keys()

date = symbol_data["timestamp"]
print(len(date))

date2 = []
for i in range(len(date)):
    date2.append(datetime.utcfromtimestamp(int(date[i]/1000)))

price = symbol_data["close"]

shortprice = []
shortdate = []
for i in range(len(date)):
    shortprice.append(price[i])
    shortdate.append(date2[i])

price3 = []
for i in range(len(shortprice)):
    if(shortprice[i] != None):
        price3.append(shortprice[i])
        exist = i
    else:
        price3.append(shortprice[exist])


print(len(price3))

print(shortdate[0], shortdate[len(shortdate)-1])

logreturn = np.zeros(len(price3))
for i in range(len(price3)-1):
    logreturn[i] = np.log(price3[i+1])-np.log(price3[i])

#plt.plot(date2, logreturn)
#plt.show()

para = estpar(logreturn)
print(para)

k = np.zeros(1001)
phi_real = np.zeros(1001)
estphi_real = np.zeros(1001)
k[0] = -500.0
for i in range(1000):
    k[i+1] = k[i] + 1.0
    phi_real[i] = phi(logreturn,float(k[i])).imag
    estphi_real[i] = estphi(float(k[i]),float(para[0]),float(para[1]),float(para[2]),float(para[3])).imag
phi_real[1000] = phi(logreturn,float(k[1000])).imag
estphi_real[1000] = estphi(float(k[100]),float(para[0]),float(para[1]),float(para[2]),float(para[3])).imag

plt.plot(k, phi_real, color='c', label='data')
plt.plot(k, estphi_real, color='b', label='estimated paramater')
plt.title("Nikkei 225")
plt.xlabel("k")
plt.ylabel("Imaginary Part of Characteristic Function")
plt.ylim(-0.25,0.25)
plt.legend()
plt.show()
