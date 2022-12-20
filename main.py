# Basics
import pandas as pd
import numpy as np

import functions.functions as functions

# Load data
print('Loading data...')
dataset = pd.read_csv('dataset/US_Accidents_dataset.csv')
print('Data loaded')

# Count None
none_by_column = dataset.isnull().sum()

# Elbow for data
plot = functions.kmeans_elbow_function(dataset[['Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng']], 2, 14)
plot.show()

# No hay codo claro... además creo que para datos geográficos quizá haya otro metodo de clustering mejor

None