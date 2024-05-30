from datetime import datetime
import csv

with open('gdaxi200803-201009para.csv', 'r') as f:
    reader = csv.reader(f)
    l = [row for row in reader]

timedif = 0

price_date = []
price = []
for i in range(len(l[0])):
    price_date.append(datetime.strptime(l[0][i], '%Y-%m-%d %H:%M:%S'))
    price.append(float(l[2][i]))

para_date = []
alpha = []
beta = []
for i in range(len(l[1])):
    para_date.append(datetime.strptime(l[1][i], '%Y-%m-%d %H:%M:%S'))
    alpha.append(float(l[3][i]))
    beta.append(float(l[4][i]))

# for i in range(timedif):
#     del price_date[0]
#     del price[0]
#     del para_date[0]
#     del alpha[0]
#     del beta[0]

maxprice = 0
for i in range(len(price)):
    if(price[i] > maxprice):
        maxprice = price[i]
        maxpriceday = i

minprice = 1000000
for i in range(len(price)):
    if(price[i] < minprice):
        minprice = price[i]
        minpriceday = i

maxalpha = 0
for i in range(len(alpha)):
    if(alpha[i] > maxalpha):
        maxalpha = alpha[i]
        maxalphaday = i

minbeta = 1000000
for i in range(len(beta)):
    if(beta[i] < minbeta):
        minbeta = beta[i]
        minbetaday = i

print(price_date[minpriceday], para_date[maxalphaday], para_date[minbetaday])

# price_lag = []
# price_date_lag = []
# para_date_lag = []
# for i in range(len(price)-timedif):
#     price_lag.append(price[timedif+i])
#     price_date_lag.append(price_date[timedif+i])
#     para_date_lag.append(para_date[timedif+i])

# alpha_lag = []
# beta_lag = []
# for i in range(len(alpha)-timedif):
#     alpha_lag.append(alpha[timedif-mindaydif+i])
#     beta_lag.append(beta[timedif-mindaydif+i])
    

# with open('gspclehmanmintimedif.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(price_date_lag)
#     writer.writerow(para_date_lag)
#     writer.writerow(price_lag)
#     writer.writerow(alpha_lag)
#     writer.writerow(beta_lag)