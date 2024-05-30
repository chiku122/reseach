from GetPara import get_para
import csv
import numpy as np
import random as rd
from scipy import stats

def stable_random(alpha, beta, gamma, delta):
    N = 100000
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

    return x

df = get_para('N225', 20040101, 20240101)

rands = [[] for i in range(len(df))]
for i in range(len(df)):
    rands[i] = stable_random(df['alpha'][i], df['beta'][i], df['gamma'][i], df['delta'][i])

with open('./data/N225/N=7828/20040101-20240101/stable_random_ab_N7828.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for i in range(len(df)):
        writer.writerow(rands[i])

print('finished')
