import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 例: 数列のデータ
data = np.random.beta(2, 5, size=1000)  # ベータ分布に従うサンプルデータ

# ベータ分布のパラメータを推定 (最大尤度推定)
alpha, beta, loc, scale = stats.beta.fit(data, floc=0, fscale=1)

print(f"推定されたパラメータ: alpha={alpha}, beta={beta}")

# 理論的なベータ分布に基づいて期待値を計算
bins = np.linspace(0, 1, 11)  # ヒストグラムのビンの数
observed_freq, _ = np.histogram(data, bins=bins, density=True)

# 理論的な頻度
x = (bins[:-1] + bins[1:]) / 2  # ビンの中心を計算
expected_freq = stats.beta.pdf(x, alpha, beta)

# 観測頻度の合計と期待頻度の合計が一致するようにスケーリング
observed_sum = np.sum(observed_freq) * (bins[1] - bins[0])  # ヒストグラムのbin幅を考慮
expected_sum = np.sum(expected_freq) * (bins[1] - bins[0])  # 同じくbin幅を考慮
observed_freq_scaled = observed_freq * observed_sum / np.sum(observed_freq)
expected_freq_scaled = expected_freq * observed_sum / expected_sum

# カイ二乗適合度検定の計算
chi2_stat, p_value = stats.chisquare(observed_freq_scaled, expected_freq_scaled)

# 結果を表示
print(f"カイ二乗統計量: {chi2_stat}")
print(f"p値: {p_value}")

# 検定の結果に基づいて判断
if p_value < 0.05:
    print("データはベータ分布に従っていないと仮定されます。")
else:
    print("データはベータ分布に従っていると仮定されます。")

# ヒストグラムと理論分布をプロット
plt.hist(data, bins=20, density=True, alpha=0.6, color='g', label='Observed Data')
plt.plot(x, expected_freq, 'r-', lw=2, label='Expected Beta Distribution')
plt.legend()
plt.show()
