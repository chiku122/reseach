import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from GetPara import get_para
import PrepareTestData

p, d, q = 10, 2, 6

arima = sm.tsa.SARIMAX(train.values,
                       order=(p, d, q)
                       ).fit(maxiter=1000)

pred_val = arima.forecast(test_size)

pred_test = pd.Series(pred_val * scale + min_val, 
                      index=test.index)

fig, ax, bx = my_plot_results(pred_test)

ax[0].set_title(f"ARIMA{(p, d, q)}")
bx[0].set_title("Root Mean Square Error")
plt.show()