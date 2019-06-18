import gym
import optuna
import pandas as pd
import numpy as np

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2

from env.TradingEnv import TradingEnv
from util.indicators import add_indicators


curr_idx = -1
reward_strategy = 'sortino'
input_data_file = 'data/SPX10min.csv'

df = pd.read_csv(input_data_file)
df = add_indicators(df.reset_index())

test_len = int(len(df) * 0.2)
train_len = int(len(df)) - test_len

train_df = df[:train_len]
test_df = df[train_len:]

train_env = DummyVecEnv([lambda: TradingEnv(
    train_df, reward_func=reward_strategy, forecast_len=50, confidence_interval=0.95)])

test_env = DummyVecEnv([lambda: TradingEnv(
    test_df, reward_func=reward_strategy, forecast_len=50, confidence_interval=0.95)])

if curr_idx == -1:
    model = PPO2(MlpLnLstmPolicy, train_env, verbose=0, nminibatches=1,
            tensorboard_log="./tensorboard")
else:
    model = PPO2.load('./agents/ppo2_' + reward_strategy + '_' + str(curr_idx) + '.pkl', env=train_env)

for idx in range(curr_idx + 1, 10):
    print('[', idx, '] Training for: ', train_len, ' time steps')

    model.learn(total_timesteps=train_len)

    obs = test_env.reset()
    done, reward_sum = False, 0

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = test_env.step(action)
        reward_sum += reward

    print('[', idx, '] Total reward: ', reward_sum, ' (' + reward_strategy + ')')
    model.save('./agents/ppo2_' + reward_strategy + '_' + str(idx) + '.pkl')
