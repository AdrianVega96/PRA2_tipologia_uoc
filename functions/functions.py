# Basics
import pandas as pd
import numpy as np

# Clustering
from sklearn import cluster

# Graphs
import matplotlib
import matplotlib.pyplot as plt

# Preprocessing
from sklearn import datasets
from sklearn import preprocessing

# Correlation visualization
import seaborn as sns

def kmeans_elbow_function(data, kmin, kmax):
    y = []
    for n in range(kmin, kmax+1):
        model = cluster.KMeans(n_clusters=n)
        # Obtengo un array con las distancias de cada elemento a cada cluster
        dists = model.fit_transform(data)
        # Sumo las distancias mínimas de cada elemento. Obtengo un array con la suma
        # de las distancias mínimas al cuadrado (clúster asignado)
        y.append(np.sum(np.min(dists, axis=1) ** 2))
    plt.figure(figsize=(16, 8))
    plt.plot(range(kmin, kmax+1), y)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method showing the optimal k')
    return plt

