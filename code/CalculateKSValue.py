from GetPara import get_para
import csv
import numpy as np
import random as rd
from scipy import stats
import matplotlib.pyplot as plt

row_num = 1212
with open('./data/N225/N=7828/20040101-20240101/stable_random_ab.csv', 'r') as f:
    reader = csv.reader(f)
    for _ in range(row_num-1):
        next(reader)
    specified_row = next(reader)
    print(specified_row)

    f.seek(0)
    reader = csv.reader(f)
    ks_value = []
    for row in reader:
        ks_value.append(stats.ks_2samp(specified_row, row).statistic)

with open('./data/N225/N=7828/20040101-20240101/ksvalue_ab(20081001).csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(ks_value)

plt.plot(ks_value)
plt.show()