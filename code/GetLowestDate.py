from GetPara import get_para

time_series = get_para('N225', 20080101, 20100101)
print(time_series[time_series['alpha'] == min(time_series['alpha'])].index)
print(time_series[time_series['price'] == min(time_series['price'])].index)

time_series = get_para('N225', 20200101, 20220101)
print(time_series[time_series['alpha'] == min(time_series['alpha'])].index)
print(time_series[time_series['price'] == min(time_series['price'])].index)

time_series = get_para('GSPC', 20080101, 20100101)
print(time_series[time_series['alpha'] == min(time_series['alpha'])].index)
print(time_series[time_series['price'] == min(time_series['price'])].index)

time_series = get_para('GSPC', 20200101, 20220101)
print(time_series[time_series['alpha'] == min(time_series['alpha'])].index)
print(time_series[time_series['price'] == min(time_series['price'])].index)

time_series = get_para('FTSE', 20080101, 20100101)
print(time_series[time_series['alpha'] == min(time_series['alpha'])].index)
print(time_series[time_series['price'] == min(time_series['price'])].index)

time_series = get_para('FTSE', 20200101, 20220101)
print(time_series[time_series['alpha'] == min(time_series['alpha'])].index)
print(time_series[time_series['price'] == min(time_series['price'])].index)

time_series = get_para('GDAXI', 20080101, 20100101)
print(time_series[time_series['alpha'] == min(time_series['alpha'])].index)
print(time_series[time_series['price'] == min(time_series['price'])].index)

time_series = get_para('GDAXI', 20200101, 20200701)
print(time_series[time_series['alpha'] == min(time_series['alpha'])].index)
print(time_series[time_series['price'] == min(time_series['price'])].index)