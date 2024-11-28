import csv
from scipy import stats
import matplotlib.pyplot as plt

row_num = 1
with open('./data/N225/stable_random/x/20040101-20240601.csv', 'r') as f:
    reader = csv.reader(f)
    for _ in range(row_num-1):
        next(reader)
    specified_row = next(reader)
    print(specified_row)

    f.seek(0)
    reader = csv.reader(f)
    ks_value = []
    for row in reader:
        ks_value.append(stats.ks_2samp(specified_row, row).statistic)

    # reader = csv.reader(f)
    # l = [row for row in reader]
    # ks_value = [0]
    # for i in range(1,len(l)):
    #     ks_value.append(stats.ks_2samp(l[i-1], l[i]).statistic)

with open('./data/N225/ksvalue/x/20040101-20240601(20040101).csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(ks_value)

plt.plot(ks_value)
plt.show()