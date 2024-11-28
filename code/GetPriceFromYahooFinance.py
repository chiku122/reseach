import csv
import yfinance as yf

ticker_symbol = '^FTSE'
stock = yf.Ticker(ticker_symbol)

df_main = stock.history(start='2004-01-01', end='2024-01-01')
df_main = df_main.interpolate()
df_for_calculate = stock.history(start='1974-01-01', end='2024-01-01')
df_for_calculate = df_for_calculate.interpolate()

price_main = list(df_main['Close'])
date_main = list(df_main.index)
price_for_calculate = list(df_for_calculate['Close'])
date_for_calulate = list(df_for_calculate.index)

N = len(date_for_calulate) - len(date_main) + 1
print(N)

with open('./data/FTSE/price/20040101-20240101.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(price_main)
    writer.writerow(date_main)
    writer.writerow(price_for_calculate)
    writer.writerow(date_for_calulate)