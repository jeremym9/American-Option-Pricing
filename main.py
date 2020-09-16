from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from env.simEnv import Equity
from env.SimRNG import Normal
from env.optionEnv import OptionsEnv
import pandas as pd
import tensorflow as tf
import os
import warnings
warnings.filterwarnings('ignore')
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['KMP_WARNINGS'] = 'off'

os.chdir('C:/Users/jeree/PycharmProjects/AmericanOptionsEnv/data')
AAPL = pd.read_csv('AAPL.csv')
AAPL.name = 'Apple'

# Experiment design parameters
ir = .025
numSamplePaths = 1000
optionLength = [21, 63, 252]
theta_k = 1

for o in optionLength:

    numSamplePaths = numSamplePaths * 2

    U = []
    for i in range(0, numSamplePaths * o):
        U.append(Normal(0, 1, 1))

    numObservables = o
    profits, day_exercised, fair_prices, naive_prices, dates = ([] for _ in range(5))

    # Create object using AAPL data
    obj = Equity(AAPL, o, numSamplePaths, numObservables, ir)

    for d in (obj.get_params()['Start Date'][:-1]):

        # Simulate data for a given date and option length
        sim_df = obj.simulate(date=d, rvs=U)
        real = obj.get_real(date=d)[numObservables:]

        # Get fair option price with the simulated data
        fair_price, naive_price = obj.get_prices(sim_df)
        fair_prices.append(fair_price)
        naive_prices.append(naive_price)

        estimated_theta_k = fair_price
        rel_error = abs(theta_k - estimated_theta_k) / abs(theta_k)
        theta_k = naive_price

        # The algorithm requires a vectorized environment to run
        train_env = DummyVecEnv([lambda: OptionsEnv(sim_df, numObservables, ir, train=True)])

        print('\nDate: ', d)
        print('Initial Price: ', real[0])
        dates.append(d)

        # Model is trained via Proximal Policy Optimization
        model = PPO2(MlpPolicy, train_env, verbose=0)
        model.learn(total_timesteps=numSamplePaths*o)
        train_env.close()

        # Out of sample testing
        test_env = DummyVecEnv([lambda: OptionsEnv(sim_df, numObservables, ir, train=False)])
        obs = test_env.reset()
        exercised = False
        day = 1
        while not exercised:
            action, states = model.predict(obs)
            obs, reward, done, info = test_env.step(action)
            if done:
                exercised = True
                print('Day ', day)
                day_exercised.append(day)
                print('Profit: ', float(reward))
                print('Fair price: ', fair_price)
                print('Naive price: ', naive_price)
                print('Rel error: ', rel_error)
                profits.append(float(reward))
            day += 1
        test_env.close()

    data = {'Dates': dates,
            'Fair Prices': fair_prices,
            'Naive Premium': naive_prices,
            'RL Premium': profits,
            'Day exercised': day_exercised}
    df = pd.DataFrame(data)
    df.name = 'MoreSims SO Option Length ' + str(o)
    df.to_csv(df.name, sep='\t')













