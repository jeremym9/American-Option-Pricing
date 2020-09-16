import math
import gym
from abc import ABC
from gym import spaces
import pandas as pd
import numpy as np
from env.SimRNG import Uniform


class OptionsEnv(gym.Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, numObservations, interest_rate, train):
        super(OptionsEnv, self).__init__()

        # Initialize environment
        self.train = train
        self.df = df
        self.numObservations = numObservations
        self.current_sim = 'Sim 1'
        self.current_step = 0
        self.strike_price = self.df[self.current_sim].iloc[self.numObservations]
        self.gamma = math.exp(-interest_rate / len(self.df.index))
        self.reward_range = (0, pd.Series.max(self.df.max(axis=None)))

        # Actions of the format: exercise, don't exercise
        self.action_space = spaces.Box(
            low=np.array([0]), high=np.array([2]), dtype=np.float16)

        # Price values of the last n observations
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1, self.numObservations+1), dtype=np.float16)

    def next_observation(self):

        # Get the price data from the last n periods and scale
        obs = np.array([
            self.df.loc[
                self.current_step: self.current_step + self.numObservations, self.current_sim].values])
        return obs

    def step(self, action):

        current_price = self.df.loc[self.current_step + self.numObservations, self.current_sim]
        reward = 0
        done = False

        if action <= 1:  # CHOOSE NOT TO EXERCISE
            reward = 0
            done = False

        elif action <= 2:  # CHOOSE TO EXERCISE
            reward = max(current_price - self.strike_price, 0) * pow(self.gamma, self.current_step)
            done = True

        self.current_step += 1

        if self.current_step + self.numObservations == len(self.df.index):
            reward = max(current_price - self.strike_price, 0) * pow(self.gamma, self.current_step)
            done = True

        obs = self.next_observation()

        return obs, reward, done, {}

    def reset(self):

        if self.train:
            randsim = int(Uniform(1, len(self.df.columns) - 1, 1))
            self.current_sim = 'Sim '+str(randsim)
        else:
            self.current_sim = 'Real'
        self.current_step = 0

        return self.next_observation()







