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

year = 14.5
short = 740

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
    logreturn1[i+1] = np.log(pricefirst[i+1])-np.log(pricefirst[i])



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
    logreturn2[i+1] = np.log(pricesecond[i+1])-np.log(pricesecond[i])


period = 10
logreturn1short = []
logreturn2short = []
for i in range(period):
    logreturn1short.append(float(logreturn1[i]))
    logreturn2short.append(float(logreturn2[i]))

c_trans = []
for i in range(len(logreturn1)-period):
    c = np.dot(logreturn1short, logreturn2short) / period
    c_trans.append(float(c))
    logreturn1short.append(float(logreturn1[period+i]))
    logreturn2short.append(float(logreturn2[period+i]))
    del logreturn1short[0]
    del logreturn2short[0]


pricefirst2 = []
for i in range(len(logreturn1)-period):
    pricefirst2.append(pricefirst[period+i])

pricesecond2 = []
for i in range(len(logreturn1)-period):
    pricesecond2.append(pricesecond[period+i])

date3 = []
for i in range(len(logreturn1)-period):
    date3.append(date2[period+i])

print(date3[0], date3[len(date3)-1])


minimum = c_trans[0]
for i in range(len(c_trans)):
    if(c_trans[i] < minimum):
        minimum = c_trans[i]
        minnum = i

mindate = date3[minnum]
print(mindate)


maximum = c_trans[0]
for i in range(len(c_trans)):
    if(c_trans[i] > maximum):
        maximum = c_trans[i]
        maxnum = i

maxdate = date3[maxnum]
print(maxdate)



fig = plt.figure()
ax1 = fig.add_subplot(111)
ln1=ax1.plot(date3, c_trans, color='y', label='C')

ax2 = ax1.twinx()
ln2= ax2.plot(date3, pricefirst2, color='steelblue', linestyle='dashed', alpha=0.2, label='Dentsu')
ln2= ax2.plot(date3, pricesecond2, color='darkgreen', linestyle='dashed', alpha=0.2, label='Nintendo')

ax1.axvline(x=14137, linestyle="--")
ax1.axvline(x=15044, linestyle="--")


fig.autofmt_xdate()

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2)

ax1.set_xlabel('date')
ax1.set_ylabel('C')
ax2.set_ylabel('stock prices')
#ax1.set_xlim(15040, 15050)
# ax1.set_ylim(0.9,1.0)
#ax2.set_ylim(0,20000)
plt.title("Dentsu, Nintendo")

plt.show()

