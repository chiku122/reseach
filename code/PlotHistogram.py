import matplotlib.pyplot as plt
import random as rd
import numpy as np
import csv
from datetime import datetime
from scipy import stats

N = 7828

def stable_random(alpha, beta, gamma, delta):
    rd.seed(0)
    v = [rd.uniform(-np.pi/2.0,np.pi/2.0) for i in range(N)]
    w = [rd.expovariate(1.0) for i in range(N)]

    b = np.arctan(beta*np.tan(np.pi*alpha/2.0))/alpha
    s = (1+beta**2.0*np.tan(np.pi*alpha/2.0)**2.0)**(1.0/(2.0*alpha))

    x = np.zeros(N)
    y = np.zeros(N)
    if(alpha-1.0 < 0.01):
        for i in range(N):
            x[i] = 2.0/np.pi*((np.pi/2.0+beta*v[i])*np.tan(v[i])-beta*np.log(np.pi/2.0*w[i]*np.cos(v[i])/(np.pi/2.0+beta*v[i])))
            y[i] = gamma*x[i]+2.0/np.pi*beta*gamma*np.log(gamma)+delta
    else:       
        for i in range(N):
            x[i] = s*np.sin(alpha*(v[i]+b))/(np.cos(v[i]))**(1.0/alpha)*(np.cos(v[i]-alpha*(v[i]+b))/w[i])**((1.0-alpha)/alpha)
            y[i] = gamma*x[i]+delta

    return y


# y = stable_random(1.8530316504855509, -0.13920823270935118, 0.009467176533904303, -0.0003464083056451395)
y = stable_random(1.5369014418944988, -0.14704825329694587, 0.005565355607241781, -0.00018134319232283842)


with open('./data/N225/200401-202401_N7828_data.csv', 'r') as f:
    reader = csv.reader(f)
    l = [row for row in reader]

date_cal = []
price_cal = []
for i in range(len(l[2])):
    date_cal.append(datetime.strptime(l[3][i], '%Y-%m-%d %H:%M:%S'))
    price_cal.append(float(l[2][i]))

logreturn = np.zeros(len(date_cal))
for i in range(len(price_cal)-1):
    logreturn[i+1] = np.log(price_cal[i+1])-np.log(price_cal[i])

r = []
for i in range(N):
    r.append(logreturn[i])

norm_random = stats.norm.rvs(size=N, scale=0.02)

tail_percentile = 25
percentile_y = np.percentile(y, tail_percentile)
percentile_r = np.percentile(r, tail_percentile)
percentile_n = np.percentile(norm_random, tail_percentile)
tail_y = [i for i in y if (i < percentile_y ) ]
tail_r = [i for i in r if (i < percentile_r )]
tail_n = [i for i in norm_random if percentile_n]

ks_s_yr, ks_p_yr = stats.ks_2samp(tail_y, tail_r)
ks_s_nr, ks_p_nr = stats.ks_2samp(tail_n, tail_r)
print(ks_s_yr, ks_p_yr)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_title('stable distribution')
ax2.set_title('emperical distribution')
ax1.hist(y, bins=100, range=(-0.1, 0.1))
ax2.hist(r, bins=100, range=(-0.1, 0.1))
plt.show()