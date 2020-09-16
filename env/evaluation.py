import os
import glob
import pandas as pd

os.chdir('C:/Users/jeree/PycharmProjects/AmericanOptionsEnv/data')
files = glob.glob('*')
timeline = ['1998', '2003', '2007', '2013', '2018']

print('Generalization performance:\n')
for f in files:
    if f == 'AAPL.csv':
        continue

    df = pd.read_csv(f, sep='\t')
    df['Numerator'] = df['RL Premium'] - df['Fair Prices']
    df['Denominator'] = df['Naive Premium'] - df['Fair Prices']

    print('For: ', f)

    print('Alpha: ', df['Numerator'].mean() / df['Denominator'].mean())
    print('Average day of exercise: ', df['Day exercised'].mean(), '\n')




