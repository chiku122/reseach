import numpy as np
import math
import scipy
import pandas as pd
import sys
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

with open('./data/parameter/GDAXI/20191218-20200618para.csv', 'r') as f:
    reader = csv.reader(f)
    l = [row for row in reader]

price_date = []
price = []
for i in range(len(l[0])):
    price_date.append(datetime.strptime(l[0][i], '%Y-%m-%d %H:%M:%S').date())
    price.append(float(l[2][i]))

para_date = []
alpha = []
beta = []
for i in range(len(l[1])):
    para_date.append(datetime.strptime(l[1][i], '%Y-%m-%d %H:%M:%S').date())
    alpha.append(float(l[3][i]))
    beta.append(float(l[4][i]))

minprice = 1000000
for i in range(len(price)):
    if(price[i] < minprice):
        minprice = price[i]
        shockday = i
        shockdate = price_date[i]

print(shockdate)

alphanorm = [0]*len(alpha)
for i in range(len(alpha)):
    alphanorm[i] = (alpha[i]-min(alpha))/(max(alpha)-min(alpha))
    if(alphanorm[i] == 1.0):
        maxalphaday = i
    if(alphanorm[i] == 0):
        minalphaday= i

betanorm = [0]*len(beta)
for i in range(len(beta)):
    betanorm[i] = (beta[i]-min(beta))/(max(beta)-min(beta))
    if(betanorm[i] == 1.0):
        maxbetaday = i
    if(betanorm[i] == 0):
        minbetaday= i

datenorm = []
start = para_date[0]
end = para_date[-1]
pd = end - start
j = 0
for i in range(pd.days+1):
    if(start + dt.timedelta(days=i) == para_date[j]):
        datenorm.append(i)
        j += 1

for i in range(len(datenorm)):
    datenorm[i] = datenorm[i]/datenorm[-1]

for i in range(len(price)):
    if(price_date[i] == dt.datetime(2020,2,18).date()):
        shockstart = i

for i in range(len(price)):
    if(price_date[i] == dt.datetime(2020,4,17).date()):
        shockend = i

datenorm = np.array(datenorm)
datenorm = datenorm.reshape([-1,1])
alphanorm = np.array(alphanorm)
alphanorm = alphanorm.reshape([-1,1])
date1 = datenorm[:shockstart]
date2 = datenorm[shockstart:shockend]
date3 = datenorm[shockend:]
alpha1 = alphanorm[:shockstart]
alpha2 = alphanorm[shockstart:shockend]
alpha3 = alphanorm[shockend:]


reg_lr = LinearRegression()
reg_lr.fit(date1, alpha1)
reg1 = reg_lr.predict(date1)
print(reg_lr.coef_, reg_lr.score(date1, alpha1))

reg_lr.fit(date2, alpha2)
reg2 = reg_lr.predict(date2)
print(reg_lr.coef_, reg_lr.score(date2, alpha2))

reg_lr.fit(date3, alpha3)
reg3 = reg_lr.predict(date3)
print(reg_lr.coef_, reg_lr.score(date3, alpha3))

regall = np.concatenate([reg1,reg2,reg3])

pricerate = [0]
for i in range(1,len(price)):
    pricerate.append(pricerate[i-1] + np.log(price[i]/price[i-1]))

fig = plt.figure()
ax1 = fig.add_subplot(111)

ln1=ax1.plot(para_date, alphanorm, color='r', label='alpha')
# ln1=ax1.plot(para_date, betanorm, color='b', label='beta')
ln1=ax1.plot(para_date, regall, color='orange', label='predict')

ax1.axvline(x=shockdate, linestyle="--")

ax2 = ax1.twinx()
ln2=ax2.plot(price_date, price, color='g', label='stock prices', linestyle='dashed', alpha=0.4)

fig.autofmt_xdate()

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2)

ax1.set_xlabel('date')
ax1.set_ylabel('alpha')
ax2.set_ylabel('stock prices')
#ax1.set_ylim(1.5,2.0)
#ax2.set_ylim(0,20000)
# ax1.axvspan(dt.datetime(2013,5,22), dt.datetime(2013,6,12), alpha=0.2)
# ax1.axvspan(dt.datetime(2008,2,4), dt.datetime(2008,3,31), alpha=0.2)
# ax1.axvspan(dt.datetime(2008,6,5), dt.datetime(2008,7,31), alpha=0.2)
plt.title("DAX")
plt.show()