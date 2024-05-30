from datetime import datetime
import csv

with open('gspc200403-201303para.csv', 'r') as f:
    reader = csv.reader(f)
    l = [row for row in reader]

timedif = 1000

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

price_norm = [0]*len(price)
for i in range(len(price)):
    price_norm[i] = (price[i]-min(price))/(max(price)-min(price))
    
beta_norm = [0]*len(beta)
for i in range(len(beta)):
    beta_norm[i] = (beta[i]-min(beta))/(max(beta)-min(beta))

period = 60
price_rate = [0]*len(price_norm)
for i in range(len(price_norm)-period):
    price_rate[i+period] = price_norm[i+period]-price_norm[i]

beta_rate = [0]*len(beta_norm)
for i in range(len(beta_norm)-period):
    beta_rate[i+period] = beta_norm[i+period]-beta_norm[i]

minprice = 1000000
for i in range(len(price_rate)):
    if(price_rate[i] < minprice):
        minprice = price_rate[i]
        minpriceday = i

minbeta = 1000000
for i in range(len(beta_rate)):
    if(beta_rate[i] < minbeta):
        minbeta = beta_rate[i]
        minbetaday = i

mindaydif = minpriceday - minbetaday

print(mindaydif, price_date[minpriceday], para_date[minbetaday])

price_lag = []
price_date_lag = []
para_date_lag = []
for i in range(len(price)-timedif):
    price_lag.append(price[timedif+i])
    price_date_lag.append(price_date[timedif+i])
    para_date_lag.append(para_date[timedif+i])

alpha_lag = []
beta_lag = []
for i in range(len(alpha)-timedif):
    alpha_lag.append(alpha[timedif-mindaydif+i])
    beta_lag.append(beta[timedif-mindaydif+i])
    



with open('gspclehmandectimedif.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(price_date_lag)
    writer.writerow(para_date_lag)
    writer.writerow(price_lag)
    writer.writerow(alpha_lag)
    writer.writerow(beta_lag)