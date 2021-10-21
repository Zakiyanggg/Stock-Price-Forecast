import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
import quandl
import csv
import scipy
from scipy.stats import chisquare

quandl.ApiConfig.api_key = '2htei_bXDXNztz2zxM2K'

# get the table for daily stock prices and,
# filter the table for selected tickers, columns within a time range
# set paginate to True because Quandl limits tables API to 10,000 rows per call

#stock = input('Please enter the code of the stock \n')
stock = 'AAPL'
with open('stock.csv', 'w', newline='') as f:
    writer = csv.writer(f)

data = quandl.get_table('WIKI/PRICES', ticker=[stock],
                        qopts={'columns': ['ticker', 'date', 'adj_close', 'adj_open']},
                        date={'gte': '2013-12-31', 'lte': '2017-12-28'},
                        paginate=True)
data.head()

closing = data.adj_close
opening = data.adj_open
dates = data.date

list1 = []
with open('stock.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for x in range(0, 1000):
        writer.writerow([closing[x], closing[x + 1], closing[x + 2], closing[x + 3], closing[x + 4], opening[x + 5]])
for x in range(0, 1000):
    list1.append([closing[x], closing[x + 1], closing[x + 2], closing[x + 3], closing[x + 4], opening[x + 5]])

dates = [1, 2, 3, 4, 5]


def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))  # convert to 1xn dimension
    x = np.reshape(x, (len(x), 1))

    svr_lin = SVR(kernel='linear', C=1e3, gamma=0.1)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2, gamma=0.1)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    # Fit regression model
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]


# predicted_price = predict_prices(dates, prices, [5])

with open('stock.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
    sixthVar = []
    predictLinSixth = []
    predictPolySixth = []
    predictRbfSixth = []
    for i in data:
        tempList = pd.to_numeric(i)
        firstFive = [None] * 5
        for j in range(5):
            firstFive[j] = tempList[j]
        sixthVar.append(tempList[5])
        predictrbfSixth, predictlinSixth, predictpolySixth = predict_prices(dates, firstFive, [5])
        predictLinSixth.append(predictlinSixth)
        predictPolySixth.append(predictpolySixth)
        predictRbfSixth.append(predictrbfSixth)

dates = list(range(1, 201))

predictRbfSixth.reverse()
predictLinSixth.reverse()
predictPolySixth.reverse()
sixthVar.reverse()
rbfpredicted_values = scipy.array(predictRbfSixth)
linpredicted_values = scipy.array(predictLinSixth)
polypredicted_values = scipy.array(predictPolySixth)
expected_values = scipy.array(sixthVar)
print('RBF goodness of fit: {0} \n'.format(scipy.stats.chisquare(rbfpredicted_values, f_exp=expected_values)))
print('Poly goodness of fit: {0} \n'.format(scipy.stats.chisquare(polypredicted_values, f_exp=expected_values)))
print('Linear goodness of fit: {0} \n'.format(scipy.stats.chisquare(linpredicted_values, f_exp=expected_values)))

plt.plot(dates, predictRbfSixth[799:999], c='g', label='RBF model')
plt.plot(dates, predictPolySixth[800:1000], c='r', label='Poly model')
plt.plot(dates, predictLinSixth[800:1000], c='b', label='Linear model')
plt.plot(dates, sixthVar[800:1000], c='y', label='Read price')
plt.xlabel('Day')
plt.ylabel('Price')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
