# Basics
import pandas as pd

import functions.Analysis as Analysis
import functions.Limpieza as Limpieza

from sklearn.model_selection import cross_val_score, train_test_split
import seaborn as sns
import statsmodels.api as sm

# Load data
print('Loading data...')
dataset = pd.read_csv(r"dataset\US_Accidents_dataset.csv")
print('Data loaded')

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
dataset.to_csv("test.csv")


############################################################################################################
#                        4.1. Selección de los grupos de datos que se quieren analizar/comparar                                  #                                                                           #
############################################################################################################
# Compute the correlation matrix
df_new = dataset[['Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
       'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)','Precipitation(in)', 'Weather_Condition']]

# Calculamos la matriz de correlación para las variables numéricas y las visualizamos con la librería seaborn
df_numerical = df_new[['Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
       'Visibility(mi)', 'Wind_Speed(mph)','Precipitation(in)']]

df_normalizado = Analysis.mean_norm(df_numerical)

corr = df_normalizado.corr()

sns.heatmap(corr)

corr = pd.DataFrame(corr)

# ANOVAs
print('ANOVAs for "Wind_Direction"')
Analysis.multiple_anovas(df_new, 'Wind_Direction', 'Distance(mi)')
Analysis.multiple_anovas(df_new, 'Wind_Direction', 'Temperature(F)')
Analysis.multiple_anovas(df_new, 'Wind_Direction', 'Wind_Chill(F)')
Analysis.multiple_anovas(df_new, 'Wind_Direction', 'Humidity(%)')
Analysis.multiple_anovas(df_new, 'Wind_Direction', 'Pressure(in)')
Analysis.multiple_anovas(df_new, 'Wind_Direction', 'Visibility(mi)')
Analysis.multiple_anovas(df_new, 'Wind_Direction', 'Wind_Speed(mph)')
Analysis.multiple_anovas(df_new, 'Wind_Direction', 'Precipitation(in)')
print('ANOVAs for "Weather_Condition"')
Analysis.multiple_anovas(df_new, 'Weather_Condition', 'Distance(mi)')
Analysis.multiple_anovas(df_new, 'Weather_Condition', 'Temperature(F)')
Analysis.multiple_anovas(df_new, 'Weather_Condition', 'Wind_Chill(F)')
Analysis.multiple_anovas(df_new, 'Weather_Condition', 'Humidity(%)')
Analysis.multiple_anovas(df_new, 'Weather_Condition', 'Pressure(in)')
Analysis.multiple_anovas(df_new, 'Weather_Condition', 'Visibility(mi)')
Analysis.multiple_anovas(df_new, 'Weather_Condition', 'Wind_Speed(mph)')
Analysis.multiple_anovas(df_new, 'Weather_Condition', 'Precipitation(in)')

# Cramer's V for correlation
confusion_matrix = pd.crosstab(dataset['Weather_Condition'], dataset['Wind_Direction'])
Analysis.cramers_v(confusion_matrix)

# Elimino la columna Wind_Chill al estar altamente correlacionada con Temperature
X = df_normalizado[['Distance(mi)', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
       'Visibility(mi)', 'Wind_Speed(mph)','Precipitation(in)']]
y = dataset['Severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 5)

# Multinomial Logistic Regression
logit_model = sm.MNLogit(y_train, sm.add_constant(X_train))
logit_model
result = logit_model.fit()
stats = result.summary()
print(stats)

print('End')

#Plot accidents
#plt= functions.plot_accidents(dataset)
#plt.show()

# Elbow for data
#plot = functions.kmeans_elbow_function(dataset[['Start_Lat', 'Start_Lng']], 10, 40)
#plot.show()

# No hay codo claro... quizá haya otro metodo de clustering mejor
# Estoy probando con más clústers en la función de elbow para ver si se ve algo.
# Si no se puede probar con clústering jerarquico.

# Habría que reducir dimensionalidad y tratar valores ausentes que se ven calculan antes
# Como hay muchisimos datos se podrían omitir las entradas que tengan valores ausentes
# en las variables de interés.

# Al graficar los accidentes se ve que hay punto bastante separados del resto que
# si estudiamos los outliers seguramente lo sean.


#None

############ Preguntas
#- ¿Factores que influyen en la severidad?
#-¿Puntos geograficos/condados de mayor riesgo?

#agrupacion de severidad basada en geografia, en condiciones climatologicas.
#Correlacionar la severidad con los tiempo de accidentes
#¿Se producen mas accidentes adelantando o no?

############# Notas:
# Quitar numbers,Timezone,Airport_Code,Give_Way,No_Exit,Station,Roundabout,Railway,Amenity,Bump,Crossing,Traffic_Calmingç
#Stop,Traffic_Signal,Turning_Loop,Sunrise_Sunset,City,conty,Description
#No sabemos: Street

#Processing
#Vlores nulos, hacer aproximacion en los datos de Precipitation(in),Wind_Speed(mph)
