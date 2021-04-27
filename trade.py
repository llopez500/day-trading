import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
import math
import tensorflow as tf
import logging
import sys
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gym
from stable_baselines import DQN, PPO2, A2C, ACKTR, DDPG, TD3, SAC

tf.get_logger().setLevel(logging.ERROR)

INITIAL_ACCOUNT_BALANCE = 100000
STOCK_DIM = 10
STOCKS = ["AAPL", "MSFT", "AMZN", "FB", "TSLA", "JPM", "NVDA", "UNH", "HD", "ADBE"]
KEY = 'Tpk_dfaeb42015194ef68ce4088f2c3b17cc'

def trade(obs, total_time):
    start = end = time.time()
    while end - start < 3600*total_time:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        obs = next(obs, getStockPrices())
        print(obs)
        print('----------------------------------------')
        end = time.time()
    env.render()


def getStockPrices():
  daily_prices = []
  symbols = ",".join(STOCKS)
  batch_api_url = f'https://sandbox.iexapis.com/stable/stock/market/batch?symbols={symbols}&types=quote&token={sys.argv[4]}'
  data = requests.get(batch_api_url).json()
  for symbol in STOCKS:
    daily_prices.append(data[symbol]['quote']['latestPrice'])
  return np.array(daily_prices)


def reset(env, balance, prices, shares=[0]*STOCK_DIM):
    obs = np.zeros(1 + 2 * STOCK_DIM)
    obs[0] = balance
    obs[1: STOCK_DIM + 1] = prices
    obs[STOCK_DIM + 1:] = shares
    env.reset(obs)

    return obs

def next(obs, prices):
    obs[1: STOCK_DIM + 1] = prices

    return obs



model = PPO2.load(f'./models/{sys.argv[1]}')
env = gym.make('gym_trade:trade-v0')
obs = reset(env, float(sys.argv[2]), getStockPrices())
trade(obs, float(sys.argv[3]))
