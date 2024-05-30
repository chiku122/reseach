import matplotlib.pyplot as plt
from GetPara import get_para
from sklearn import preprocessing
from MakeSteady import make_steady

def plot_data(stock_index, start_date, end_date, param):
    time_series = get_para(stock_index, start_date, end_date)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    fig.autofmt_xdate()
    ax1.set_xlabel('date')

    if(param == 'alpha'):
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
    if(param == 'both'):
        time_series['alpha'] = preprocessing.minmax_scale(time_series['alpha'])
        time_series['gamma'] = preprocessing.minmax_scale(time_series['gamma'])
        ln1 = ax1.plot(time_series['alpha'], color='r', label='alpha')
        ln1 = ax1.plot(time_series['gamma'], color='b', label='beta')
        ax1.set_ylabel('parameter')
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
    if(param == 'ksvalue_and_price'):
        ln1 = ax1.plot(time_series.index, time_series['ksvalue'], color='purple', label='ksvalue')
        ax1.set_ylabel('ks value')
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

plot_data('N225', 20040101, 20240101, 'ksvalue_and_price')