import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt

from sklearn import cluster

import numpy as np

# Preprocessing
from sklearn import datasets
from sklearn import preprocessing

# Correlation visualization
import seaborn as sns

def show_clusters_over_map(dtf_X, model):
    # Get cluster size
    count = dtf_X.groupby(["cluster"]).count().reset_index()
    si = count["Start_Lat"].apply(lambda x: (10000 * x) / count["Start_Lat"].sum()).tolist()

    # plot
    si = count["Start_Lat"].apply(lambda x: (1000 * x) / count["Start_Lat"].sum()).tolist()
    fig = go.Figure(data=go.Scattergeo(
        lon=model.cluster_centers_[:, 0],
        lat=model.cluster_centers_[:, 1],
        mode='markers',
        marker={"size": si}
    ))

    fig.update_layout(
        title='US accident focus',
        geo_scope='usa',
    )
    fig.show()


def kmeans_elbow_function(data, kmin, kmax):
    y = []
    for n in range(kmin, kmax + 1):
        model = cluster.KMeans(n_clusters=n)
        # Obtengo un array con las distancias de cada elemento a cada cluster
        dists = model.fit_transform(data)
        # Sumo las distancias mínimas de cada elemento. Obtengo un array con la suma
        # de las distancias mínimas al cuadrado (clúster asignado)
        y.append(np.sum(np.min(dists, axis=1) ** 2))
    plt.figure(figsize=(16, 8))
    plt.plot(range(kmin, kmax + 1), y)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method showing the optimal k')
    return plt

def plot_accidents(dataset):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(dataset['Start_Lng'], dataset['Start_Lat'], s=8, alpha=.1, c=dataset["Severity"])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.axis('equal')
    plt.tight_layout()
    return plt