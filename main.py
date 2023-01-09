# Basics
import pandas as pd

import functions.Analysis as Analysis
import functions.Limpieza as Limpieza
import functions.Visualization as Visualization

from sklearn.model_selection import cross_val_score, train_test_split
import seaborn as sns
import statsmodels.api as sm
from sklearn import cluster

import plotly.graph_objects as go

# Load data
print('Loading data...')
dataset = pd.read_csv(r"dataset\US_Accidents_dataset.csv")
print('Data loaded')

plot = Visualization.plot_accidents(dataset)
plot.show()

##################################### 2.Integración y selección ###############################################
print('Comienza la Integración y selección ...')
dataset = Limpieza.SelectColumns(dataset)
dataset = Limpieza.ColumnTransform(dataset)
##################################### 3.Limpieza de los datos ###############################################

############################################################################################################
#            3.1. ¿Los datos contienen ceros o elementos vacíos? Gestiona cada uno de estos casos.         #                                                                           #
############################################################################################################
print('Comienza la limpieza de NaNs ...')
dataset = Limpieza.CleanNan(dataset)

############################################################################################################
#                        3.2. Identifica y gestiona los valores extremos                                   #                                                                           #
############################################################################################################
print('Comienza el tratamiento de Outliers ...')
dataset = Limpieza.CleanOutlier(dataset)
dataset = Limpieza.Sampling(dataset)
dataset = Limpieza.RemoveColumns(dataset)

dataset.to_csv('.\dataset\dataset.csv')

############################################################################################################
#                        4.1. Selección de grupos                                                          #                                                                           #
############################################################################################################

df = dataset[['Severity', 'Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
       'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition']]

df_clustering = dataset[['Start_Lng', 'Start_Lat']]

df['Severity'] = df['Severity'].astype('category')

############################################################################################################
#                        4.2. Comprobar normalidad y homocedasticidad                                      #                                                                           #
############################################################################################################

s = Analysis.sampletest(df)
print(s)

df_var = Analysis.SampleHomocedasticidad(df)
print(df_var)

############################################################################################################
#                        4.3. Pruebas estadísticas                                                         #                                                                           #
############################################################################################################

# Variables para calcular matriz de correlación
df_numerical = df[['Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
       'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']]

df_normalizado = Analysis.mean_norm(df_numerical)

# Correlation matrix
corr = df_normalizado.corr()

sns.heatmap(corr)

corr = pd.DataFrame(corr)

# ANOVAs
print('ANOVAs for "Wind_Direction"')
Analysis.multiple_anovas(df, 'Wind_Direction', 'Distance(mi)')
Analysis.multiple_anovas(df, 'Wind_Direction', 'Temperature(F)')
Analysis.multiple_anovas(df, 'Wind_Direction', 'Wind_Chill(F)')
Analysis.multiple_anovas(df, 'Wind_Direction', 'Humidity(%)')
Analysis.multiple_anovas(df, 'Wind_Direction', 'Pressure(in)')
Analysis.multiple_anovas(df, 'Wind_Direction', 'Visibility(mi)')
Analysis.multiple_anovas(df, 'Wind_Direction', 'Wind_Speed(mph)')
Analysis.multiple_anovas(df, 'Wind_Direction', 'Precipitation(in)')
print('ANOVAs for "Weather_Condition"')
Analysis.multiple_anovas(df, 'Weather_Condition', 'Distance(mi)')
Analysis.multiple_anovas(df, 'Weather_Condition', 'Temperature(F)')
Analysis.multiple_anovas(df, 'Weather_Condition', 'Wind_Chill(F)')
Analysis.multiple_anovas(df, 'Weather_Condition', 'Humidity(%)')
Analysis.multiple_anovas(df, 'Weather_Condition', 'Pressure(in)')
Analysis.multiple_anovas(df, 'Weather_Condition', 'Visibility(mi)')
Analysis.multiple_anovas(df, 'Weather_Condition', 'Wind_Speed(mph)')
Analysis.multiple_anovas(df, 'Weather_Condition', 'Precipitation(in)')

# Cramer's V for correlation
confusion_matrix = pd.crosstab(dataset['Weather_Condition'], dataset['Wind_Direction'])
print(Analysis.cramers_v(confusion_matrix))

# Elimino la columna Wind_Chill al estar altamente correlacionada con Temperature
X = df_normalizado[['Distance(mi)', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
       'Visibility(mi)', 'Wind_Speed(mph)','Precipitation(in)']]
y = dataset['Severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 5)

# Multinomial Logistic Regression
logit_model = sm.MNLogit(y_train, sm.add_constant(X_train))
result = logit_model.fit()
stats = result.summary()
print(stats)

# K-means
# Elbow for data
plot2 = Visualization.kmeans_elbow_function(dataset[['Start_Lat', 'Start_Lng']], 10, 100)
plot2.show()

#Model
k = 50
model = cluster.KMeans(n_clusters=k, init='k-means++')
# Select variables
X = df_clustering

# clustering
dtf_X = X.copy()
dtf_X["cluster"] = model.fit_predict(X)

############################################################################################################
#                        5. Representación de resultados                                                   #                                                                           #
############################################################################################################

sns.heatmap(corr)

print(stats)

Visualization.show_clusters_over_map(dtf_X, model)

print('End')

