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


year = 3.1
short = 732

my_share = share.Share('4324.T')
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

logreturn1 = np.zeros(len(pricefirst))
for i in range(len(pricefirst)-1):
    logreturn1[i+1] = np.log(pricefirst[i+1]) - np.log(pricefirst[i])



my_share = share.Share('7974.T')
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

logreturn2 = np.zeros(len(pricesecond))
for i in range(len(pricesecond)-1):
    logreturn2[i+1] = np.log(pricesecond[i+1]) - np.log(pricesecond[i])

inner = np.zeros(len(logreturn1))
for i in range(len(logreturn1)):
    inner[i] = logreturn1[i]*logreturn2[i]




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


fig = plt.figure()
ax1 = fig.add_subplot(111)
ln1=ax1.plot(shortdate, inner, color='y', label='I')

ax2 = ax1.twinx()
ln2=ax2.plot(shortdate, pricefirst, color='steelblue', linestyle='dashed', alpha=0.2, label='Dentsu')
ln2= ax2.plot(shortdate, pricesecond, color='darkgreen', linestyle='dashed', alpha=0.2, label='Nintendo')

fig.autofmt_xdate()

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2)

ax1.set_xlabel('date')
ax1.set_ylabel('C')
ax2.set_ylabel('stock prices')
# ax1.set_ylim(0.9,1.0)
#ax2.set_ylim(0,20000)
plt.title("Dentsu,Nintendo")

plt.show()

