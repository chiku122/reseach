import matplotlib.pyplot as plt
import statsmodels.api as sm
import GetPara

# コレログラム
def correlogram(time_series, lags):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(time_series, title='Autocorrelation', lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(time_series, title='Partial Autocorrelation', lags=lags, ax=ax2)
    plt.show()

time_series = GetPara.get_para('N225', 20190301, 20210301)
correlogram(time_series['beta'], str(240))