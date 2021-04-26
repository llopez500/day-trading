import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gym
from gym import error, spaces, utils
from gym.utils import seeding

STOCK_DIM = 10
INITIAL_ACCOUNT_BALANCE = 100000
SHARE_SCALING = 1000

class StockTrainEnv(gym.Env):
  """A stock trading environment for OpenAI gym"""
  metadata = {'render.modes': ['human']}

  def __init__(self, df):
    super(StockTrainEnv, self).__init__()

    self.time = 0
    #daily stock prices (time by price)
    # time = np.linspace(1.5*np.pi, 10.5*np.pi, 200)
    # price = np.sin(time) + 2
    #
    # stocks = ["AAPL", "MSFT", "AMZN", "FB", "TSLA", "JPM", "NVDA", "UNH", "HD", "ADBE"]
    # df = pd.DataFrame()
    # for stock in stocks:
    #     df[stock] = price

    self.df = df
    #stock prices on given time
    self.data = self.df.loc[self.time, :]

    #actions sell, hold, or buy.
    self.action_space = spaces.Box(
        low = -1, high = 1,shape = (STOCK_DIM,))

    #current balance, stock prices, and owned share
    self.observation_space = spaces.Box(
        low=0, high=np.inf, shape = (2*STOCK_DIM + 1,))

    #initialize state
    self.state = np.array([INITIAL_ACCOUNT_BALANCE] + list(self.data) + [0]*STOCK_DIM)
    self.portfolio_value = [INITIAL_ACCOUNT_BALANCE]
    self.done = False

  def _buy(self, index, shares):
      balance = self.state[0]
      price = self.state[index + 1]

      max_possible_shares = balance//price
      shares_bought = min(max_possible_shares, shares)

      self.state[index + STOCK_DIM + 1] += shares_bought
      self.state[0] -= shares_bought*price

  def _sell(self, index, shares):
      balance = self.state[0]
      price = self.state[index + 1]

      max_possible_shares = self.state[index + STOCK_DIM + 1]
      shares_sold = min(max_possible_shares, abs(shares))

      self.state[index + STOCK_DIM + 1] -= shares_sold
      self.state[0] += shares_sold*price

  def reset(self):
    #intialize at a random time
    # self.time = randint(0, len(self.df.index))

    #initialize at the beginning
    self.time = 0
    self.data = self.df.loc[self.time, :]
    self.state = np.array([INITIAL_ACCOUNT_BALANCE] + list(self.data) + [0]*STOCK_DIM)
    self.portfolio_value = [INITIAL_ACCOUNT_BALANCE]

    return self.state

  def step(self, actions):
    #upscale the number of shares bought/sold bye SHARE_SCALING
    shares = actions*SHARE_SCALING

    prev_portfolio_value = self.state[0] + sum((self.state[1: STOCK_DIM + 1])*(self.state[STOCK_DIM + 1: ]))
    # self.data = self.df.loc[self.time, :]
    # self.state[1:STOCK_DIM + 1] = list(self.data)
    # reward = (self._totalAssets() - self.totalAssets)
    # self.profit += reward
    # reward = reward*1e-1

    #update the balance and number of shares based on actions
    sell_indices = [idx for idx, val in enumerate(shares) if val < 0]
    buy_indices = [idx for idx, val in enumerate(shares) if val > 0]
    for idx in sell_indices: self._sell(idx, shares[idx])
    for idx in buy_indices: self._buy(idx, shares[idx])

    #update the stock prices
    self.time += 1
    self.data = self.df.loc[self.time, :]
    self.state[1:STOCK_DIM + 1] = list(self.data)

    cur_portfolio_value = self.state[0] + sum((self.state[1: STOCK_DIM + 1])*(self.state[STOCK_DIM + 1: ]))
    self.portfolio_value.append(cur_portfolio_value)

    # print(sell_indices, buy_indices, shares, cur_portfolio_value)

    #assign return values
    observation = self.state
    reward = cur_portfolio_value - prev_portfolio_value
    self.done = True if self.time >= len(self.df.index) - 1 else False
    info = {}

    return observation, reward, self.done, info

  def render(self, mode='human', close=False):
    plt.plot(self.portfolio_value)
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.title("Change in Portfolio Value Over Time")
    plt.show()
