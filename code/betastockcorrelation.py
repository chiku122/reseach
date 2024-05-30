import numpy as np
import math
import scipy
import pandas as pd
import sys
from datetime import datetime
import csv
import matplotlib.pyplot as plt


with open('n225lehmanrecoverypara.csv', 'r') as f:
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


timedif = 50
period = 10

for i in range(timedif-period+1):
    del price_date[0]
    del price[0]
    del para_date[0]
    del alpha[0]
    del beta[0]

betalogreturn = np.zeros(len(beta))
for i in range(len(beta)-1):
    betalogreturn[i+1] = np.log(-beta[i+1])-np.log(-beta[i])

pricelogreturn = np.zeros(len(price))
for i in range(len(price)-1):
    pricelogreturn[i+1] = np.log(price[i+1])-np.log(price[i])


betawindow = []
pricewindow = []
for i in range(period):
    betawindow.append(float(betalogreturn[i]))
    pricewindow.append(float(pricelogreturn[i]))

c = []
for i in range(len(betalogreturn)-period):
    cwindow = -np.dot(betawindow, pricewindow) / period
    c.append(float(cwindow))
    betawindow.append(float(betalogreturn[period+i]))
    pricewindow.append(float(pricelogreturn[period+i]))
    del betawindow[0]
    del pricewindow[0]

for i in range(period):
    del price_date[0]
    del price[0]
    del para_date[0]
    del alpha[0]
    del beta[0]


betanorm = [0]*len(beta)
for i in range(len(beta)):
    betanorm[i] = (beta[i]-min(beta))/(max(beta)-min(beta))

pricenorm = [0]*len(price)
for i in range(len(price)):
    pricenorm[i] = (price[i]-min(price))/(max(price)-min(price))


fig = plt.figure()
ax1 = fig.add_subplot(111)
ln1=ax1.plot(para_date, c, color='y', label='C')

ax2 = ax1.twinx()
ln2= ax2.plot(para_date, betanorm, color='b', linestyle='dashed', alpha=0.2, label='beta')
ln2= ax2.plot(price_date, pricenorm, color='g', linestyle='dashed', alpha=0.2, label='stock prices')

fig.autofmt_xdate()

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2)

ax1.axhline(y=0, linestyle="--")

ax2.tick_params(right=False, labelright=False)

ax1.set_xlabel('date')
ax1.set_ylabel('C')
plt.title("Nikkei 225")

plt.show()

