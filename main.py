import pandas as pd
import numpy as np

# Load data
print('Loading data...')
dataset = pd.read_csv('dataset/US_Accidents_dataset.csv')
print('Data loaded')

# Count None
none_by_column = dataset.isnull().sum()

None