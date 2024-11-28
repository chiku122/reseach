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

# alpha, beta, gamma, delta = estpar(data)
# print(alpha, beta, gamma, delta)
# y = stable_random(alpha, beta, gamma, delta)
# print(y)
# plt.hist(y, bins=1000, density=True, alpha=0.6, color='g', label='Observed Data')
# # plt.xlim(0, 2)
# plt.show()


plt.hist(data, bins=200, density=True, alpha=0.6, color='g', label='Observed Data')

# ベータ分布のパラメータ（α, β）を指定
a, b = 0.2, 10

# # ベータ分布の確率密度関数をプロット
# x = np.linspace(0, 1, 1000)
# pdf = stats.beta.pdf(x, a, b)
# plt.plot(x, pdf, 'r-', lw=2, label='Beta PDF')

# # KS検定
# D, p_value = stats.kstest(data, 'beta', args=(a, b))
# print(f"KS検定: D={D}, p-value={p_value}")

# # # Q-Qプロットを作成
# stats.probplot(data, dist="beta", sparams=(a, b), plot=plt)


plt.show()
