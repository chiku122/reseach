from fastdtw import fastdtw
import numpy as np
import math
import scipy
import pandas as pd
import sys
from datetime import datetime
import csv
import matplotlib.pyplot as plt

with open('ftsecovidrecoverypara.csv', 'r') as f:
    reader = csv.reader(f)
    l = [row for row in reader]

timedif = 50

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

for i in range(timedif):
    del price_date[0]
    del price[0]
    del para_date[0]
    del alpha[0]
    del beta[0]


price_norm = [0]*len(price)
for i in range(len(price)):
    price_norm[i] = (price[i]-min(price))/(max(price)-min(price))

beta_norm = [0]*len(beta)
for i in range(len(beta)):
    beta_norm[i] = (beta[i]-min(beta))/(max(beta)-min(beta))

distance = 0
path = 0
mindistance = 1000000
for i in range(len(beta_norm)-50):
    price_window = [price_norm[j] for j in range(i+50)]
    beta_window = [beta_norm[j] for j in range(i+50)]
    distance, path = fastdtw(price_window, beta_window)
    distperday = distance / len(beta_window)

    if(distperday < mindistance):
        mindistance = distperday
        mindistday = len(beta_window)
        mindistprice = price_window
        mindistbeta = beta_window
        minpath = path

print(mindistday, mindistance)
print(len(minpath), len(mindistbeta), len(mindistprice))

plt.plot(mindistprice, color="g", label="price")
plt.plot(mindistbeta, color="b", label="beta")
plt.title("FTSE 100")

for a_x, b_x in minpath:
  plt.plot([a_x, b_x], [mindistprice[a_x], mindistbeta[b_x]], color='gray', linestyle='dotted', linewidth=1)
  
plt.legend()
plt.show()