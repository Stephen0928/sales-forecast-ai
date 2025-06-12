import pandas as pd

df = pd.read_csv("Sales Dataset.csv")
print(df['Product Category'].unique())