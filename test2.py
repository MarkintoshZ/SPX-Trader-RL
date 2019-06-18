import gym
import optuna
import pandas as pd

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2

from env.TradingEnv import TradingEnv
from util.indicators import add_indicators

curr_idx = 0
reward_strategy = 'sortino'
input_data_file = 'data/SPX10min.csv'

df = pd.read_csv(input_data_file)
# df = df.sort_values(['Date'])
df = add_indicators(df.reset_index())

test_len = int(len(df) * 0.2)
train_len = int(len(df)) - test_len

test_df = df[train_len:]

test_env = DummyVecEnv([lambda: TradingEnv(
    test_df, reward_func=reward_strategy, forecast_len=50, confidence_interval=0.95)])

model = PPO2.load('./agents/ppo2_' + reward_strategy + '_' + str(curr_idx) + '.pkl', env=test_env)

obs, done = test_env.reset(), False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = test_env.step(action)

    test_env.render(mode="human")
