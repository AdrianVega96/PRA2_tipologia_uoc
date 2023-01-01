# Basics
import pandas as pd
import functions.functions as functions
import functions.Limpieza as Limpieza

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

#print('End')
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
