from fastdtw import fastdtw
import numpy as np
import math
import scipy
import pandas as pd
import sys
from datetime import datetime
import csv
import matplotlib.pyplot as plt


with open('n225covidrecoverypara.csv', 'r') as f:
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
price_interval = 1
para_interval = 1

price_window = []
for i in range(len(price)-timedif):
    price_window.append(price[i+timedif])

price_norm = [0]*len(price_window)
for i in range(len(price_window)):
    price_norm[i] = (price_window[i]-min(price_window))/(max(price_window)-min(price_window))

distance = 0
path = 0
mindistance = 1000000
for i in range(timedif):
    beta_window = []
    for j in range(len(beta)-timedif):
        beta_window.append(beta[i+j])
    
    beta_norm = [0]*len(beta_window)
    for j in range(len(beta_window)):
        beta_norm[j] = (beta_window[j]-min(beta_window))/(max(beta_window)-min(beta_window))
    
    distance, path = fastdtw(price_norm, beta_norm)
    if(distance < mindistance):
        mindistance = distance
        mintimedif = timedif - i
        minbeta_norm = beta_norm

print(mintimedif, mindistance)
print(0, distance)
print(len(path), len(price_norm))

plt.plot(minprice_norm, color="g", label="price")
plt.plot(minbeta_norm, color="b", label="beta")
plt.title("Nikkei 225")

for a_x, b_x in path:
  plt.plot([a_x, b_x], [price_norm[a_x], beta_norm[b_x]], color='gray', linestyle='dotted', linewidth=1)
  
plt.legend()
plt.show()