# Basics
import pandas as pd

from sklearn.cluster import DBSCAN

import functions.functions as functions

# Load data
print('Loading data...')
dataset = pd.read_csv('dataset/US_Accidents_dataset.csv')
print('Data loaded')

# Count None
none_by_column = dataset.isnull().sum()

# Plot accidents
#plt= functions.plot_accidents(dataset)
#plt.show()

# Elbow for data
plot = functions.kmeans_elbow_function(dataset[['Start_Lat', 'Start_Lng']], 10, 40)
plot.show()

# No hay codo claro... quizá haya otro metodo de clustering mejor
# Estoy probando con más clústers en la función de elbow para ver si se ve algo.
# Si no se puede probar con clústering jerarquico.

# Habría que reducir dimensionalidad y tratar valores ausentes que se ven calculan antes
# Como hay muchisimos datos se podrían omitir las entradas que tengan valores ausentes
# en las variables de interés.

# Al graficar los accidentes se ve que hay punto bastante separados del resto que
# si estudiamos los outliers seguramente lo sean.

print('End')
None