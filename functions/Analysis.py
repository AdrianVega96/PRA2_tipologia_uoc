# Basics
import pandas as pd
import numpy as np
import scipy.stats as ss
from scipy import stats
import pingouin as pg

# Clustering
from sklearn import cluster

# Graphs
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


def NormalTest(data, nombre, pl=1):
    statistic, p_value = stats.kstest(data, 'norm', N=100, args=(np.mean(data), np.std(data, ddof=1)))

    # Interpretación
    normal = False

    alpha = 0.05
    if pl == 1:
        if p_value > alpha:
            # print(nombre)
            print('La muestra es Normal. No se rechaza la hipótesis nula H0')
            normal = True
        else:
            # print(nombre)
            print('La muestra no es Normal. Se rechaza la hipótesis nula H0')
            normal = False

    return statistic, p_value, normal


def sampletest(df):
    dfs = df.sample(n=100, replace=True)
    df_input = dfs.select_dtypes(include=np.number)
    s = pd.DataFrame([NormalTest(df_input[column], column, pl=0) for column in df_input],
                     columns=["Stats", "P-value", "Normal"])
    s["Variable"] = [column for column in df_input]


def TestHomocedasticidad(data, nombre):
    # Levene's Test in Python using Pingouin
    S = pg.homoscedasticity(data, dv=nombre, group='Severity', center="median")
    return S.loc["levene", "W"], S.loc["levene", "pval"], S.loc["levene", "equal_var"]

    # s1=data.loc[data['Severity']==1,nombre].tolist()
    # s2=data.loc[data['Severity']==2,nombre].tolist()
    # s3=data.loc[data['Severity']==3,nombre].tolist()
    # s4=data.loc[data['Severity']==4,nombre].tolist()
    # stat, p = bartlett(s1, s2, s3,s4)
    # alpha=0.5
    # normal=False
    # if p>alpha:
    #    normal=True
    # else:
    #    normal=False
    # return stat, p, normal


def SampleHomocedasticidad(df):
    dfs = df  # df.sample(n=1000,replace=True)
    var_name = [column for column in dfs.select_dtypes(include=np.number)]
    variables = var_name + ['Severity']
    df_input = dfs[variables]
    s = [TestHomocedasticidad(df_input.loc[:, ['Severity', column]], column) for column in var_name]
    df_var = pd.DataFrame(s, columns=["W", "P-value", "equal_var"])
    # df_var=pd.DataFrame(s,columns=["stat","p","equal_var"])
    df_var["Variable"] = var_name

    return df_var

