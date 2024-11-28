import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import csv
from EstimateParam import estpar, stable_random

# 例としてランダムに生成したデータ（実際にはここにあなたのデータが入ります）
with open('./data/N225/numbers.csv', 'r') as f:
    reader = csv.reader(f)
    l = [row for row in reader]

data = []
for i in range(len(l[0])):
    data.append(float(l[0][i]))

Q1 = np.percentile(data, 25)  # 第1四分位数 (25パーセンタイル)
Q3 = np.percentile(data, 75)  # 第3四分位数 (75パーセンタイル)
IQR = Q3 - Q1  # 四分位範囲

# 外れ値の閾値
lower_bound = Q1 - 50 * IQR
upper_bound = Q3 + 50 * IQR

# 外れ値を検出
outliers = [x for x in data if x < lower_bound or x > upper_bound]

print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
print(f"Outliers: {outliers}")