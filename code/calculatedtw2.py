from fastdtw import fastdtw
import numpy as np
import math
import scipy
import pandas as pd
import sys
from datetime import datetime
import csv
import matplotlib.pyplot as plt


with open('gdaxi20200318-20210318para.csv', 'r') as f:
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

price_norm = [0]*len(price)
for i in range(len(price)):
    price_norm[i] = (price[i]-min(price))/(max(price)-min(price))

beta_norm = [0]*len(beta)
for j in range(len(beta)):
    beta_norm[j] = (beta[j]-min(beta))/(max(beta)-min(beta))

distance, path = fastdtw(price_norm, beta_norm)

print(distance)

plt.plot(price_norm, color="g", label="price")
plt.plot(beta_norm, color="b", label="beta")
plt.title("Nikkei 225")

for a_x, b_x in path:
  plt.plot([a_x, b_x], [price_norm[a_x], beta_norm[b_x]], color='gray', linestyle='dotted', linewidth=1)
  
plt.legend()
plt.show()