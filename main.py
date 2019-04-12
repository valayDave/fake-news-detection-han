import pandas as pd
import train as model

dataset_path = 'datasets/first_split.csv'

df = pd.read_csv(dataset_path)
model.train(df,'first_split_dataset_han')
