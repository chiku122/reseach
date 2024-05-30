import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
import GetPara

def seasonal_decompose(time_series, period):
    res = sm.tsa.seasonal_decompose(time_series, period=period)
    fig = res.plot()
    fig.autofmt_xdate()
    plt.show()

def plot_stl(time_series, period, highlight_date):
    stl = STL(time_series, period=period, robust=True).fit()
    
    fig, ax = plt.subplots(4, 1, figsize=(7, 5), sharex=True)

    time_series.plot(ax=ax[0], c='black')
    ax[0].set_title("Original")
    ax[0].axvline(x=highlight_date, linestyle="--")

    stl.trend.plot(ax=ax[1], c='black')
    ax[1].set_title("Trend")
    ax[1].axvline(x=highlight_date, linestyle="--")

    stl.seasonal.plot(ax=ax[2], c='black')
    ax[2].grid(True, linestyle='--', alpha=0.5)
    ax[2].set_title("Seasonal")
    ax[2].axvline(x=highlight_date, linestyle="--")

    stl.resid.plot(ax=ax[3], c='black')
    ax[3].set_title("Residual")
    ax[3].axvline(x=highlight_date, linestyle="--")

    plt.tight_layout()

    plt.show()