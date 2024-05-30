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


year = 6.1
short = 1486

my_share = share.Share('9432.T')
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

pricefirst = []
for i in range(len(shortprice)):
    if(shortprice[i] != None):
        pricefirst.append(shortprice[i])
        exist = i
    else:
        pricefirst.append(shortprice[exist])

print(len(pricefirst))

ratio1 = np.zeros(len(pricefirst)-1)
for i in range(len(pricefirst)-1):
    ratio1[i] = np.log(pricefirst[i+1]) - np.log(pricefirst[i])



my_share = share.Share('9433.T')
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

pricesecond = []
for i in range(len(shortprice)):
    if(shortprice[i] != None):
        pricesecond.append(shortprice[i])
        exist = i
    else:
        pricesecond.append(shortprice[exist])

print(len(pricesecond))

ratio2 = np.zeros(len(pricesecond)-1)
for i in range(len(pricesecond)-1):
    ratio2[i] = np.log(pricesecond[i+1]) - np.log(pricesecond[i])


period_fix = 750
inner_trans = [[],[],[],[],[],[]]
for j in range(6):
    period = period_fix - 125*j
    ratio1short = []
    ratio2short = []
    for i in range(period):
        ratio1short.append(float(ratio1[i+125*j]))
        ratio2short.append(float(ratio2[i+125*j]))

    for i in range(len(ratio1)-period_fix):
        inner = np.dot(ratio1short, ratio2short)
        inner_trans[j].append(float(inner))
        ratio1short.append(float(ratio1[period_fix+i]))
        ratio2short.append(float(ratio2[period_fix+i]))
        del ratio1short[0]
        del ratio2short[0]

inner_ave = np.sum(inner_trans,axis=0)
inner_ave = inner_ave / 6


pricefirst2 = []
for i in range(len(ratio1)-period_fix):
    pricefirst2.append(pricefirst[period_fix+i])

pricesecond2 = []
for i in range(len(ratio1)-period_fix):
    pricesecond2.append(pricesecond[period_fix+i])

date3 = []
for i in range(len(ratio1)-period_fix):
    date3.append(date2[period_fix+i])

print(date3[0], date3[len(date3)-1])



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

pricenikkei = []
for i in range(len(shortprice)):
    if(shortprice[i] != None):
        pricenikkei.append(shortprice[i])
        exist = i
    else:
        pricenikkei.append(shortprice[exist])

print(len(pricenikkei))

pricenikkei2 = []
for i in range(len(ratio1)-period_fix):
    pricenikkei2.append(pricenikkei[period_fix+i])


fig = plt.figure()
ax1 = fig.add_subplot(111)
ln1=ax1.plot(date3,inner_ave, color='y', label='I')

ax2 = ax1.twinx()
ln2=ax2.plot(date3,pricefirst2, color='steelblue', linestyle='dashed', alpha=0.2, label='NTT')
ln2= ax2.plot(date3,pricesecond2, color='darkgreen', linestyle='dashed', alpha=0.2, label='KDDI')

fig.autofmt_xdate()

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2)

ax1.set_xlabel('date')
ax1.set_ylabel('I')
ax2.set_ylabel('stock prices')
# ax1.set_ylim(0.9,1.0)
#ax2.set_ylim(0,20000)
plt.title("NTT, KDDI")

plt.show()

