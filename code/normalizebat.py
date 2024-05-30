import numpy as np
import math
import scipy
import pandas as pd
import sys
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import datetime as dt


with open('n225201205-201405para.csv', 'r') as f:
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

alpharapid = []
for i in range(len(alpha)-1):
    if(abs(alpha[i+1]-alpha[i]) > 0.005):
        alpharapid.append(i)
alpharapid.append(len(alpha))
print(alpharapid)

alphapart = []
betapart = []
pricepart = []
alphanorm = []
betanorm = []
pricenorm = []
for i in range(alpharapid[0]):
    alphapart.append(alpha[i])
    betapart.append(beta[i])
    pricepart.append(price[i])
for i in range(len(alphapart)):
    alphanorm.append((alphapart[i]-min(alphapart))/(max(alphapart)-min(alphapart)))
    betanorm.append((betapart[i]-min(betapart))/(max(betapart)-min(betapart)))
    pricenorm.append((pricepart[i]-min(pricepart))/(max(pricepart)-min(pricepart)))

for j in range(len(alpharapid)-1):
    alphanorm.append(None)
    betanorm.append(None)
    pricenorm.append(None)
    alphapart = []
    betapart = []
    pricepart = []
    for i in range(alpharapid[j]+1, alpharapid[j+1]):
        alphapart.append(alpha[i])
        betapart.append(beta[i])
        pricepart.append(price[i])

    for i in range(len(alphapart)):
        if(len(alphapart) < 30):
            alphanorm.append(None)
            betanorm.append(None)
            pricenorm.append(None)
        else:
            alphanorm.append((alphapart[i]-min(alphapart))/(max(alphapart)-min(alphapart)))
            betanorm.append((betapart[i]-min(betapart))/(max(betapart)-min(betapart)))
            pricenorm.append((pricepart[i]-min(pricepart))/(max(pricepart)-min(pricepart)))

for i in range(len(alpha)):
    if(alphanorm[i] == None):
        print(para_date[i])

fig = plt.figure()
ax1 = fig.add_subplot(111)

# ln1=ax1.plot(para_date, alphanorm, color='r', label='alpha')
ln1=ax1.plot(para_date, betanorm, color='b', label='beta')

# ax1.axvline(x=dt.datetime(2008,9,15,0,0), linestyle="--")
# ax1.axvline(x=dt.datetime(2020,3,12,0,0), linestyle="--")

ax2 = ax1.twinx()
ln2=ax2.plot(price_date, pricenorm, color='g', label='stock prices', linestyle='dashed', alpha=0.4)

fig.autofmt_xdate()

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2)

ax1.set_xlabel('date')
ax1.set_ylabel('beta')
ax2.set_ylabel('stock prices')
# ax1.set_xlim(dt.datetime(2008,1,1), dt.datetime(2009,1,1))
# ax1.set_ylim(-0.5,1.5)
# ax2.set_ylim(-0.5,1.5)
# ax1.axvspan(dt.datetime(2020,2,21), dt.datetime(2020,6,10), alpha=0.2)
# ax1.axvspan(dt.datetime(2020,9,18), dt.datetime(2020,11,6), alpha=0.2)
# ax1.axvspan(dt.datetime(2020,9,2), dt.datetime(2021,2,24), alpha=0.2)
plt.title("Nikkei 225")

plt.show()
