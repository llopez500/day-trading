import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import sys

from stable_baselines import DQN, PPO2, A2C, ACKTR, DDPG, TD3, SAC
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv

import gym
from gym import spaces

tf.get_logger().setLevel(logging.ERROR)

training_data = pd.read_csv(f'./data/{sys.argv[1]}').iloc[:, 1:]
env = gym.make('gym_trade:train-v0', df=training_data)

model = sys.argv[2]

if model == 'None':
    model = PPO2('MlpPolicy', env, n_steps=200, nminibatches=4)
else:
    model = PPO2.load(f'./models/{model}')
    model.set_env(DummyVecEnv([lambda: gym.make('gym_trade:train-v0', df=training_data)]))

print('Training Model...')
model.learn(total_timesteps=50000)
print('Finished Training!')

model.save(f'./models/{sys.argv[3]}')
