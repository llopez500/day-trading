import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import gym
from gym import error, spaces, utils
from gym.utils import seeding

STOCK_DIM = 10
INITIAL_ACCOUNT_BALANCE = 100000
SHARE_SCALING = 1000

class StockTradeEnv(gym.Env):
  """A stock trading environment for OpenAI gym"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(StockTradeEnv, self).__init__()
    #actions sell, hold, or buy.
    self.action_space = spaces.Box(
        low = -1, high = 1, shape = (STOCK_DIM,))

    #current balance, stock prices, and owned share
    self.observation_space = spaces.Box(
        low=0, high=np.inf, shape = (2*STOCK_DIM + 1,))

    #initialize state: balance, price, shares
    self.state = np.array([INITIAL_ACCOUNT_BALANCE] + [0]*(2*STOCK_DIM))
    self.portfolio_value = [INITIAL_ACCOUNT_BALANCE]
    self.done = False

  def _buy(self, index, shares):
      balance = self.state[0]
      price = self.state[index + 1]

      if price > 0: max_possible_shares = balance//price
      else: return

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

  def reset(self, state, portfolio_value):
    #initialize at the beginning
    self.state = state
    self.portfolio_value = [portfolio_value]
    self.done = False

    return self.state

  def step(self, actions):
    #upscale the number of shares bought/sold bye SHARE_SCALING
    shares = actions*SHARE_SCALING

    prev_portfolio_value = self.portfolio_value[-1]

    #update the balance and number of shares based on actions
    sell_indices = [idx for idx, val in enumerate(shares) if val < 0]
    buy_indices = [idx for idx, val in enumerate(shares) if val > 0]
    for idx in sell_indices: self._sell(idx, shares[idx])
    for idx in buy_indices: self._buy(idx, shares[idx])

    cur_portfolio_value = self.state[0] + sum((self.state[1: STOCK_DIM + 1])*(self.state[STOCK_DIM + 1: ]))
    self.portfolio_value.append(cur_portfolio_value)

    # print(sell_indices, buy_indices, shares, cur_portfolio_value)

    #assign return values
    observation = self.state
    reward = cur_portfolio_value - prev_portfolio_value
    self.done = True if cur_portfolio_value <= 0 else False
    info = self.portfolio_value[-1]

    return observation, reward, self.done, info

  def render(self, mode='human', close=False):
    # Create figure
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=list(range(len(self.portfolio_value))), y=self.portfolio_value))

    # Set title
    fig.update_layout(
        title_text="Portfolio Value Over Time"
    )

    #Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type="linear"
        )
    )

    fig.show()
