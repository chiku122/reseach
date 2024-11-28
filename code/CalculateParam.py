import numpy as np
from datetime import datetime
import csv
from EstimateParam import estpar

########################################
# 実行
# data は array型
# estpar(data)

with open('./data/FTSE/price/20040101-20240101.csv', 'r') as f:
    reader = csv.reader(f)
    l = [row for row in reader]

date = []
price = []
for i in range(len(l[0])):
    date.append(datetime.strptime(l[1][i], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None))
    price.append(float(l[0][i]))

date_cal = []
price_cal = []
for i in range(len(l[2])):
    date_cal.append(datetime.strptime(l[3][i], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None))
    price_cal.append(float(l[2][i]))

print(len(price), len(price_cal))


logreturn = np.zeros(len(date_cal))
for i in range(len(price_cal)-1):
    logreturn[i+1] = np.log(price_cal[i+1])-np.log(price_cal[i])


N = 5053
window_dif = 250
partition = 28
alpha_trans = [[] for i in range(partition)]
beta_trans = [[] for i in range(partition)]
gamma_trans = [[] for i in range(partition)]
delta_trans = [[] for i in range(partition)]
alphaerror = [0]*partition
betaerror = [0]*partition
gammaerror = [0]*partition
for j in range(partition):
    period = N - window_dif*j
    logreturn_window = []
    for i in range(period):
        logreturn_window.append(float(logreturn[i+window_dif*j]))

    for i in range(len(logreturn)-N):
        para = estpar(logreturn_window)
        if(float(para[0]) == 0 or float(para[0]) == 2):
            alphaerror[j] += 1
        if(float(para[1]) == -1 or float(para[1]) == 1):
            betaerror[j] += 1
        if(float(para[2]) <= 0):
            gammaerror[j] += 1
        alpha_trans[j].append(float(para[0]))
        beta_trans[j].append(float(para[1]))
        gamma_trans[j].append(float(para[2]))
        delta_trans[j].append(float(para[3]))
        logreturn_window.append(float(logreturn[N+i]))
        del logreturn_window[0]

    para = estpar(logreturn_window)
    alpha_trans[j].append(float(para[0]))
    beta_trans[j].append(float(para[1]))
    gamma_trans[j].append(float(para[2]))
    delta_trans[j].append(float(para[3]))

print(alphaerror, betaerror, gammaerror)

alpha_ave = np.sum(alpha_trans,axis=0)
alpha = alpha_ave / partition

beta_ave = np.sum(beta_trans,axis=0)
beta = beta_ave / partition

gamma_ave = np.sum(gamma_trans,axis=0)
gamma = gamma_ave / partition

delta_ave = np.sum(delta_trans,axis=0)
delta = delta_ave / partition


with open('./data/FTSE/param/20040101-20240101.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(date)
    writer.writerow(price)
    writer.writerow(alpha)
    writer.writerow(beta)
    writer.writerow(gamma)
    writer.writerow(delta)