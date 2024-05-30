from fastdtw import fastdtw
import numpy as np
import math
import scipy
import pandas as pd
import sys
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import datetime as dt


with open('gdaxi200307-202307para.csv', 'r') as f:
    reader = csv.reader(f)
    l = [row for row in reader]

price_date = []
price = []
for i in range(len(l[0])):
    price_date.append(datetime.strptime(l[0][i], '%Y-%m-%d %H:%M:%S'))
    price.append(float(l[2][i]))

para_date = []
alpha = []
beta = []
for i in range(len(l[1])):
    para_date.append(datetime.strptime(l[1][i], '%Y-%m-%d %H:%M:%S'))
    alpha.append(float(l[3][i]))
    beta.append(float(l[4][i]))

period = 60
distance = [0]*(len(beta)-period)

for i in range(len(beta)-period):
    price_window = []
    for j in range(period):
        price_window.append(price[i+j])

    price_norm = [0]*len(price_window)
    for j in range(len(price_window)):
        price_norm[j] = (price_window[j]-min(price_window))/(max(price_window)-min(price_window))
    
    beta_window = []
    for j in range(period):
        beta_window.append(beta[i+j])
    
    beta_norm = [0]*len(beta_window)
    for j in range(len(beta_window)):
        beta_norm[j] = (beta_window[j]-min(beta_window))/(max(beta_window)-min(beta_window))

    for j in range(len(beta_norm)):
        distance[i] += abs(beta_norm[j] - price_norm[j])

mindistance = 1000000
for i in range(len(distance)):
    if(distance[i] < mindistance):
        mindistance = distance[i]
        mindisperiod = i

print(mindistance, price_date[mindisperiod])

shortdistance = []
shortdisperiod = []
for i in range(len(distance)):
    if(distance[i] <= 5.0):
        shortdistance.append(distance[i])
        shortdisperiod.append(i)

# for i in range(len(shortdistance)):
#     print(shortdistance[i], para_date[shortdisperiod[i]])

shortdisprice = []
shortdisbeta = []
shortdisdate = []
for i in range(period):
    shortdisprice.append(price[shortdisperiod[6]+i])
    shortdisbeta.append(beta[shortdisperiod[6]+i])
    shortdisdate.append(price_date[shortdisperiod[6]+i])

mindisprice = []
mindisbeta = []
mindisdate = []
for i in range(period):
    mindisprice.append(price[mindisperiod+i])
    mindisbeta.append(beta[mindisperiod+i])
    mindisdate.append(price_date[mindisperiod+i])


fig = plt.figure()
ax1 = fig.add_subplot(111)

# ln1=ax1.plot(para_date, alpha, color='r', label='alpha')
ln1=ax1.plot(shortdisdate, shortdisbeta, color='b', label='beta')

# ax1.axvline(x=dt.datetime(2008,9,15,0,0), linestyle="--")
# ax1.axvline(x=dt.datetime(2020,3,12,0,0), linestyle="--")

ax2 = ax1.twinx()
ln2=ax2.plot(shortdisdate, shortdisprice, color='g', label='stock prices', linestyle='dashed', alpha=0.4)

fig.autofmt_xdate()

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2)

ax1.set_xlabel('date')
ax1.set_ylabel('beta')
ax2.set_ylabel('stock prices')
#ax1.set_ylim(-0.5,0)
#ax2.set_ylim(0,20000)
plt.title("S&P 500")

plt.show()