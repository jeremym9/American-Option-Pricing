import pandas as pd
import numpy as np
import math


class Equity:

    def __init__(self, df, option_length, num_sims, num_observables, ir):

        # Initialization
        self.name = df.name
        self.numSamplePaths = num_sims
        self.numObservables = num_observables
        self.interest_rate = ir
        self.option_length = option_length
        self.df = df[['Date', 'Close']]
        self.sim_mean = 0
        self.sim_std = 0

    def get_params(self):

        params_df = pd.DataFrame(columns=['Start Date', 'Initial Price', 'Mu', 'Sigma', 'Path'])
        disposable = self.df

        # Creation of option scenarios with sliding window = option length
        for i in range(0, int(np.floor(self.df.shape[0] / self.option_length))):

            period = disposable.head(n=self.option_length*2)
            disposable = disposable.iloc[self.option_length:]

            prev = period.iloc[:self.option_length]
            actual = period.iloc[self.option_length:]

            real_path = np.append(
                prev['Close'].to_numpy()[-self.numObservables:], actual['Close'].to_numpy())
            start_date = actual['Date'].to_numpy()[0]
            initial_price = actual['Close'].to_numpy()[0]
            returns = prev['Close'].pct_change()
            mu = np.mean(np.log1p(returns))
            sigma = np.std(np.log1p(returns), ddof=1)

            params_df.loc[i] = [start_date] + [initial_price] + [mu] + [sigma] + [real_path]

        return params_df

    def get_real(self, date):

        params = self.get_params()

        # Get real asset path with n observables
        if date not in params['Start Date'].to_numpy():
            print('Invalid date')
            return
        else:
            idx = params['Start Date'][params['Start Date'] == date].index[0]
            test_data = params['Path'][idx]
            return list(test_data)

    def simulate(self, date, rvs):

        params = self.get_params()

        # Simulate sample paths for given date
        if date not in params['Start Date'].to_numpy():
            print('Invalid date')
            return
        else:
            idx = params['Start Date'][params['Start Date'] == date].index[0]
            observables = params['Path'][idx][:self.numObservables]
            drift = params['Mu'][idx]
            volatility = params['Sigma'][idx]
            df = pd.DataFrame()
            cols = []

            # Construction of dataframe containing all sample paths (replications)
            j = 0
            for n in range(self.numSamplePaths):
                X = params['Initial Price'][idx]
                value_list = [X]

                # Discretized Geometric Brownian Motion
                for s in range(self.option_length-1):
                    Z = rvs[j]
                    X = float(X * np.exp(drift + volatility * Z))
                    value_list.append(X)
                    j += 1
                df[n] = list(observables) + list(value_list)
                cols.append('Sim ' + str(n + 1))

            # Attach real data at the end
            df['Real'] = self.get_real(date)
            cols.append('Real')

            # Normalization of the form (x - mu) / sigma
            self.sim_mean = pd.Series.mean(df.mean(axis=None))
            self.sim_std = pd.Series.std(df.std(axis=None))
            df = pd.DataFrame(
                np.divide(np.subtract(df.to_numpy(), self.sim_mean), self.sim_std))

            df.columns = cols

            return df

    def get_prices(self, df):

        real = df['Real'].iloc[self.numObservables:].to_numpy()
        df = df.iloc[self.numObservables:].drop('Real', axis=1).reset_index(drop=True)
        strike_price = df['Sim 1'].iloc[0]
        fair_values = []
        naive_values = []

        # Compute fair price with in-sample simulation
        # Compute naive price with out-of-sample real path
        t = 1
        for index, row in df.iterrows():
            arr = np.multiply(np.maximum(
                    row.to_numpy() - strike_price, np.zeros(
                        self.numSamplePaths)), pow(math.exp(-self.interest_rate / len(df.index)), t))
            fair_values.append(np.mean(arr))
            naive_values.append(
                max(real[t-1] - strike_price, 0) * pow(
                    math.exp(-self.interest_rate / len(df.index)), t))
            t += 1
        fair_price = np.mean(fair_values)
        naive_price = np.mean(naive_values)

        return fair_price, naive_price
