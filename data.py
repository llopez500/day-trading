import numpy as np
import pandas as pd
import requests
import sys
import time
import math

KEY =  sys.argv[3]
#'Tpk_dfaeb42015194ef68ce4088f2c3b17cc'
STOCKS = list(pd.read_csv('./data/stocks.csv').columns)
print(STOCKS)

dataset = pd.DataFrame(columns=STOCKS)

def collectDataSet(total_time, name):
    start = end = time.time()
    while end - start < 3600*total_time:
        dataset_len = len(dataset)
        daily_prices = []
        symbols = ",".join(STOCKS)
        batch_api_url = f'https://sandbox.iexapis.com/stable/stock/market/batch?symbols={symbols}&types=quote&token={KEY}'
        data = requests.get(batch_api_url).json()
        for symbol in STOCKS:
            daily_prices.append(data[symbol]['quote']['latestPrice'])
        dataset.loc[dataset_len] = daily_prices
        end = time.time()

    dataset.to_csv(f'./data/{name}')

print('Collecting Data...')
collectDataSet(float(sys.argv[2]), sys.argv[1])
print('Succesfully collected data!')
