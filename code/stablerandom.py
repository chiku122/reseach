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



def stable_random(alpha, beta):
    N = 1000000
    rd.seed(0)
    v = [rd.uniform(-np.pi/2.0,np.pi/2.0) for i in range(N)]
    w = [rd.expovariate(1.0) for i in range(N)]

    b = np.arctan(beta*np.tan(np.pi*alpha/2.0))/alpha
    s = (1+beta**2.0*np.tan(np.pi*alpha/2.0)**2.0)**(1.0/(2.0*alpha))

    x = np.zeros(N)
    if(alpha-1.0 < 0.01):
        for i in range(N):
            x[i] = 2.0/np.pi*((np.pi/2.0+beta*v[i])*np.tan(v[i])-beta*np.log(np.pi/2.0*w[i]*np.cos(v[i])/(np.pi/2.0+beta*v[i])))

    else:       
        for i in range(N):
            x[i] = s*np.sin(alpha*(v[i]+b))/(np.cos(v[i]))**(1.0/alpha)*(np.cos(v[i]-alpha*(v[i]+b))/w[i])**((1.0-alpha)/alpha)

    return x

N = 1000000
stable = stable_random(1.5,0)
# sx = sorted(stable)
# sy = [i/N for i in range(N)]

plt.xlim(-10,10)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.hist(stable, bins=1000, range=(-10,10))
plt.show()
