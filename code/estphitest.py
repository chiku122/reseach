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

def estphi(k, alpha, beta, gamma, delta):
    p = np.exp(complex(-(gamma*k)**alpha,delta*k+(gamma*k)**alpha*beta*math.tan(math.pi*alpha/2)))
    return p

para = [1.77678472e+00,-2.05404862e-01,8.24633613e-03,-3.52740701e-05]

k = np.zeros(1001)
#phi_real = np.zeros(101)
estphi_real = np.zeros(1001)
k[0] = -500.0
for i in range(1000):
    k[i+1] = k[i] + 1.0
    #phi_real[i] = phi(logreturn,k[i]).real
    estphi_real[i] = estphi(float(k[i]),para[0],para[1],para[2],para[3]).real
#phi_real[100] = phi(logreturn,k[100]).real
estphi_real[1000] = estphi(float(k[100]),para[0],para[1],para[2],para[3]).real

print(estphi_real)
plt.plot(k, estphi_real)

#plt.xlabel("date")
#plt.ylabel("price")
plt.show()