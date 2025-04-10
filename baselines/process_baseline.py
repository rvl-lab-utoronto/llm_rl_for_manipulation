import pandas as pd

df = pd.read_csv("results.csv")
df = df[['model', 'category', 'reward', 'run_time']]
df = df.groupby(['model', 'category']).mean()

print(df)