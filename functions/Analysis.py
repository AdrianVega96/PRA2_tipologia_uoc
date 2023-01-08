# Basics
import pandas as pd
import numpy as np
import scipy.stats as ss

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

def plot_accidents(dataset):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(dataset['Start_Lng'], dataset['Start_Lat'], s=8, alpha=.1)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.axis('equal')
    plt.tight_layout()
    return plt


# Función para normalizar dataframe
def mean_norm(df_input):
    return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)


def multiple_anovas(df, cat_item, num_item):
    from scipy.stats import f_oneway
    category_list = df.groupby(cat_item)[num_item].apply(list)
    anova_result = f_oneway(*category_list)
    print(f'P-value for {cat_item} and {num_item} ANOVA is {anova_result[1]}')
    if anova_result[1] < 0.05:
        print('Rejected H0, both are correlated (95%)')
    else:
        print('Acepted H0, both are not correlated (95%)')


def cramers_v(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.to_numpy().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))