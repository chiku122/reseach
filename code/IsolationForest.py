import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from GetPara import get_para

# np.random.seed(123)

# # 平均と分散
# mean1 = np.array([2, 2])
# mean2 = np.array([-2, -2])
# cov = np.array([[1, 0], [0, 1]])

# # 正常データの生成(2つの正規分布から生成)
# norm1 = np.random.multivariate_normal(mean1, cov, size=100)
# norm2 = np.random.multivariate_normal(mean2, cov, size=100)

# # 異常データの生成(一様分布から生成)
# lower, upper = -10, 10
# anom = (upper - lower)*np.random.rand(10, 2) + lower

# df = np.vstack([norm1, norm2, anom])
# df = pd.DataFrame(df, columns=["feat1", "feat2"])

# 可視化
# sns.scatterplot(x="feat1", y="feat2", data=df)

# sklearnでの実装

df = get_para('GSPC', 20200101, 20240101)
sk_df = df[["alpha", "beta", "gamma", "delta"]]
clf = IsolationForest(n_estimators=1000, random_state=123)
clf.fit(sk_df)
pred = clf.predict(sk_df)

score = clf.decision_function(sk_df)

# グラフ作成
fig, ax = plt.subplots(1,1, tight_layout=True)

# 異常度
error_ax = ax
error_ax.plot(df.index, score, color='red')
error_ax.set_xlabel("date")
error_ax.set_ylabel("error")
error_ax.legend()
error_ax.grid()
# error_ax.axvline(test_data[test_data['error'] == d_max].index, linestyle='--')
# for i in range(len(score)):
#     if score[i] <= -0.1:
#         error_ax.axvline(df.index[i], alpha=0.2, color='red')
error_ax.tick_params(axis='x', rotation=30)

price_ax = ax.twinx()
price_ax.plot(df.index, df['price'], color='black', label='stock prices', linestyle='solid', alpha=1.0, linewidth=0.5)
# price_ax.plot(test_data.index, test_data['price'].rolling(window=50, center=True).mean(), color='yellowgreen', label='price')
price_ax.set_ylabel('stock prices')

plt.show()


# 可視化
# sns.scatterplot(x="alpha", y="beta" data=sk_df, hue='predict', palette='bright')

# plt.show()
