import sys
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import pandas as pd
import numpy as np
import datetime as dt
import csv
 
my_share = share.Share('^N225')
symbol_data = None
 
try:
    symbol_data = my_share.get_historical(
        share.PERIOD_TYPE_YEAR, 51,
        share.FREQUENCY_TYPE_DAY, 1)
except YahooFinanceError as e:
    print(e.message)
    sys.exit(1)
 
df_from_yahoo = pd.DataFrame(symbol_data)
df_from_yahoo['datetime'] = pd.to_datetime(df_from_yahoo.timestamp, unit='ms')
df_from_yahoo = df_from_yahoo.interpolate()

df_main = df_from_yahoo[(df_from_yahoo['datetime'] >= dt.datetime(2004,1,1,0,0)) & (df_from_yahoo['datetime'] <= dt.datetime(2024,3,7,0,0))]
df_for_calculate = df_from_yahoo[(df_from_yahoo['datetime'] >= dt.datetime(2001,2,16,0,0)) & (df_from_yahoo['datetime'] <= dt.datetime(2024,3,7,0,0))]

price_main = list(df_main['close'])
date_main = list(df_main['datetime'])
price_for_calculate = list(df_for_calculate['close'])
date_for_calulate = list(df_for_calculate['datetime'])

N = len(date_for_calulate) - len(date_main) + 1
print(N)

with open('./data/N225/20040101-20240307_N750_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(price_main)
    writer.writerow(date_main)
    writer.writerow(price_for_calculate)
    writer.writerow(date_for_calulate)