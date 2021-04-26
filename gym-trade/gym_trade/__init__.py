from gym.envs.registration import register

register(
    id='trade-v0',
    entry_point='gym_trade.envs:StockTradeEnv',
)
register(
    id='train-v0',
    entry_point='gym_trade.envs:StockTrainEnv',
)
