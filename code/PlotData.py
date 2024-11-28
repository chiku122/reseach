import matplotlib.pyplot as plt
from GetPara import get_para
from sklearn import preprocessing
from MakeSteady import make_steady
import numpy as np

def plot_data(stock_index, start_date, end_date, param):
    time_series = get_para(stock_index, start_date, end_date)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    fig.autofmt_xdate()
    ax1.set_xlabel('date')

    if(param == 'alpha'):
        # time_series['alpha'] = make_steady(time_series['alpha'], 1)
        ln1 = ax1.plot(time_series['alpha'], color='r', label='alpha')
        ax1.set_ylabel('alpha')
        ax1.grid()
        ax2 = ax1.twinx()
        ln2 = ax2.plot(time_series.index, time_series['price'], color='g', label='stock prices', linestyle='dashed', alpha=0.4)
        ax2.set_ylabel('stock prices')

    if(param == 'beta'):
        ln1 = ax1.plot(time_series['beta'], color='b', label='beta')
        ax1.set_ylabel('beta')

    if(param == 'gamma'):
        ln1 = ax1.plot(time_series['gamma'], color='y', label='gamma')
        ax1.set_ylabel('gamma')

    if(param == 'delta'):
        ln1 = ax1.plot(time_series['delta'], color='g', label='delta')
        ax1.set_ylabel('delta')

    if(param == 'all'):
        time_series['alpha'] = preprocessing.minmax_scale(time_series['alpha'])
        time_series['beta'] = preprocessing.minmax_scale(time_series['beta'])
        time_series['gamma'] = preprocessing.minmax_scale(time_series['gamma'])
        time_series['delta'] = preprocessing.minmax_scale(time_series['delta'])

        ln1 = ax1.plot(time_series['alpha'], color='r', label='alpha')
        ln2 = ax1.plot(time_series['beta'], color='b', label='beta')
        ln3 = ax1.plot(time_series['gamma'], color='y', label='gamma')
        ln4 = ax1.plot(time_series['delta'], color='g', label='delta')
        ax1.set_ylabel('parameters')
        ax1.grid()
        ax2 = ax1.twinx()
        ln2 = ax2.plot(time_series.index, time_series['price'], color='black', label='stock prices', linestyle='solid', alpha=1.0, linewidth=0.5)
        ax2.set_ylabel('stock prices')

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2)

    if(param == 'ksvalue'):
        ln1 = ax1.plot(time_series.index, time_series['ksvalue'], color='purple', label='ksvalue')
        ax1.set_ylabel('ks value')
        ax1.grid()
        ax2 = ax1.twinx()
        ln2 = ax2.plot(time_series.index, time_series['price'], color='g', label='stock prices', linestyle='dashed', alpha=0.4)
        ax2.set_ylabel('stock prices')

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2)

    if(param == 'price'):
        ln1 = ax1.plot(time_series['price'], color='green', label='price', alpha=0.2)
        ln1 = ax1.plot(time_series['price'].rolling(window=50, center=True).mean(), color='yellowgreen', label='price')
        ax1.set_ylabel('price')

    if(param == 'var'):
        ln1 = ax1.plot(time_series['var'], color='purple', label='var')
        ax1.axvline(time_series[time_series['var'] == max(time_series['var'])].index, linestyle='--')
        ax1.set_ylabel('var')
        ax1.grid()
        ax2 = ax1.twinx()
        ln2 = ax2.plot(time_series.index, time_series['price'], color='g', label='stock prices', linestyle='dashed', alpha=0.4)
        ax2.set_ylabel('stock prices')

    if(param == 'volatility'):
        historical_log_diff = [np.log(time_series['price'][i]/time_series['price'][i-1]) for i in range(1, len(time_series['price']))]
        # ksvalue_log_diff = [np.log(time_series['ksvalue'][i]/time_series['ksvalue'][i-1]) for i in range(2, len(time_series['ksvalue']))]
        ksvalue_log_diff = [np.log(time_series['ksvalue'][i]) for i in range(1, len(time_series['ksvalue']))]
        historical_volatility = [np.std(historical_log_diff[i-10:i]) for i in range(10, len(historical_log_diff))]
        ksvalue_volatility = [np.std(ksvalue_log_diff[i-10:i]) for i in range(10, len(ksvalue_log_diff))]
        historical_volatility = preprocessing.minmax_scale(historical_volatility)
        ksvalue_volatility = preprocessing.minmax_scale(ksvalue_volatility)
        date = time_series.index[11:]
        ln1 = ax1.plot(date, ksvalue_volatility, color='purple', label='ksvalue volatility')
        ln1 = ax1.plot(date, historical_volatility, color='blue', label='historical volatility')
        ax1.grid()
        ax2 = ax1.twinx()
        ln2 = ax2.plot(time_series.index, time_series['price'], color='g', label='stock prices', linestyle='dashed', alpha=0.4)
        ax2.set_ylabel('stock prices')

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2)
    

    # ax1.axvline(time_series[time_series['price'] == min(time_series['price'])].index, linestyle='--')

    #ax1.set_ylim(1.5,2.0)
    #ax2.set_ylim(0,20000)
    # ax1.axvspan(dt.datetime(2013,5,22), dt.datetime(2013,6,12), alpha=0.2)
    # ax1.axvspan(dt.datetime(2008,2,4), dt.datetime(2008,3,31), alpha=0.2)
    # ax1.axvspan(dt.datetime(2008,6,5), dt.datetime(2008,7,31), alpha=0.2)
    plt.title('{}'.format(stock_index))

    return(
        plt.show()
    )

plot_data('N225', 20040101, 20241001, 'alpha')