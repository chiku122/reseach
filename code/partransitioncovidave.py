# pythonによる安定分布の推定手法
# version 1.0 2020/9/20
# version 1.1 2020/9/23
# version 1.2 2020/10/24

import random as rd
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
from scipy import stats


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

def stable_random(alpha, beta):
    N = 100000
    rd.seed(0)
    v = [rd.uniform(-np.pi/2.0,np.pi/2.0) for i in range(N)]
    w = [rd.expovariate(1.0) for i in range(N)]

    b = np.arctan(beta*np.tan(np.pi*alpha/2.0))/alpha
    s = (1+beta**2.0*np.tan(np.pi*alpha/2.0)**2.0)**(1.0/(2.0*alpha))

    x = np.zeros(N)
    if(alpha-1.0 < 0.01):
        for i in range(N):
            x[i] = 2.0/np.pi*((np.pi/2.0+beta*v[i])*np.tan(v[i])-beta*np.log(np.pi/2.0*w[i]*np.cos(v[i])/(np.pi/2.0+beta*v[i])))

    else:       
        for i in range(N):
            x[i] = s*np.sin(alpha*(v[i]+b))/(np.cos(v[i]))**(1.0/alpha)*(np.cos(v[i]-alpha*(v[i]+b))/w[i])**((1.0-alpha)/alpha)

    return x
    
########################################
# 実行
# data は array型
# estpar(data)

year = 17.5
short = 1483

my_share = share.Share('^N225')
symbol_data = None

try:
    symbol_data = my_share.get_historical(share.PERIOD_TYPE_YEAR,year,
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
for i in range(short):
    shortprice.append(price[i])
    shortdate.append(date2[i])

print(shortdate[0], shortdate[len(shortdate)-1])

price3 = []
for i in range(len(shortprice)):
    if(shortprice[i] != None):
        price3.append(shortprice[i])
        exist = i
    else:
        price3.append(shortprice[exist])

print(len(price3))

logreturn = np.zeros(len(price3))
for i in range(len(price3)-1):
    logreturn[i+1] = np.log(price3[i+1])-np.log(price3[i])

period_fix = 750
alpha_trans = [[],[],[],[],[],[]]
for j in range(6):
    period = period_fix - 125*j
    logreturn2 = []
    for i in range(period):
        logreturn2.append(float(logreturn[i+125*j]))

    for i in range(len(logreturn)-period_fix):
        para = estpar(logreturn2)
        alpha_trans[j].append(float(para[0]))
        logreturn2.append(float(logreturn[period_fix+i]))
        del logreturn2[0]

alphaave = np.sum(alpha_trans,axis=0)
alphaave = alphaave / 6

price2 = []
for i in range(len(logreturn)-period_fix):
    price2.append(price3[period_fix+i])

date3 = []
for i in range(len(logreturn)-period_fix):
    date3.append(date2[period_fix+i])

print(date3[0], date3[len(date3)-1])

# for i in range(6):
#     print(len(alpha_trans[i]))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ln1=ax1.plot(date3,alphaave, color='r', label='alpha')
ax1.axvline(x=14137, linestyle="--")

ax2 = ax1.twinx()
ln2=ax2.plot(date3,price2, color='g', label='stock prices')

fig.autofmt_xdate()

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2)

ax1.set_xlabel('date')
ax1.set_ylabel('alpha')
ax2.set_ylabel('stock prices')
#ax1.set_xlim(14100,14200)
ax1.set_ylim(1.4,2.0)
#ax2.set_ylim(0,20000)
plt.title("Nikkei 225")

plt.show()