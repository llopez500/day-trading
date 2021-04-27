# Automated Stock Trading Using Deep Reinforcement Learning

This repository provides code to easily test deep reinforcement learning based automated day-trading strategies  

## Abstract

Reinforcement learning (RL) is a brach of machine learning concerned with maximizing the cumulative reward of a agent by allowing it to lern an optimal policy through its interactions with its environment. In recent year, reinforcement learning has leveraged the use of deep neural networks as function approximators. The code below provides a training/testing pipeline that facilitates the collection of data from IEX Cloud, training of deep RL models, and automated trading. 

## Installation:
```bash
git clone https://github.com/llopez500/day-trading.git
```

### Prerequisites

To collect data using this repository, you'll first need to make an account with [IEX Cloud](https://iexcloud.io/) to receive and API key. 

Install the gym training environments with the following command: 
```bash
pip install -e gym-trade
```
Install dependencies using the following command: 
```bash
pip install -r requirements.text
```

NOTE: Stable Baselines is currently only compatible with TensorFlow 1.x.

## Usage: 

To collect data, train, and trade, follow the instructions below.

### Data Collection

Enter the following command, substituting names and values in angle bracket with your own (NOTE: Make sure to have an API key for IEX Cloud first).
```bash
python data.py <save_data.csv> <hours> <API key>
```
The 'hours' is a float value indicating how long you would like to collect data. Data files are saved in data folder.

### Training

To collect data, enter the following command: 
```bash
python train.py <load_data.csv> <load_model.zip> <save_model.zip>
```
If you are training a new model, enter None for 'load_data.csv'. All models are saved in models folder.

### Trading

To do some real-time trading you can use the command below (NOTE: Make sure to have an API key for IEX Cloud first).
```bash
python trade.py <load_model.zip> <balance> <hours> <API key>
```
Here, 'balance' refers to money available to trade and 'hours' is a float representing the number of hours you would like to trade for. 