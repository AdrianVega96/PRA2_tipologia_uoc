import pandas as pd
import datetime as dt
import calendar
import numpy as np
from sklearn.utils import resample


def SelectColumns(df):

    #Drop Bools
    bool_idx=[x for x in df.dtypes!="bool"]
    df=df.iloc[:, bool_idx]

    #Drop Object
    describe = df.describe(include="object").T.sort_values(by=["unique"])
    Object_drop = describe.loc[describe["unique"] <= 3, ["freq"]].apply(
        lambda x: 100 * (x / df.shape[0])).index.tolist()
    df=df.drop(Object_drop, axis=1)

    #Drop Columns without meaning
    toDrop = ["Timezone", "ID", "Description", "County"]
    df=df.drop(toDrop, axis=1)

    #Drop numeric
    df=df.drop(["Number"], axis=1)
    return df

def mappers(df):
    horas = [list(range(x - 6, x)) for x in range(6, 30, 6)]
    rangos = ["0-5", "6-11", "12-17", "18-23"]
    mapTime = {j: i for e, i in zip(horas, rangos) for j in e}
    df['Time'] = df['Time'].map(mapTime)

    mapMonth = {m: calendar.month_name[m] for m in range(1, 13)}
    df['Month'] = df['Month'].map(mapMonth)

    mapWind = {'SW': "SW", 'WSW': "W", 'West': "W", 'NNW': "N", 'WNW': 'W', 'NW': 'NW', 'W': 'W', 'SSW': "S",
               'East': "E", 'SE': 'SE',
               'North': "N", 'ENE': 'E', 'NNE': "N", 'NE': 'NE', 'SSE': "S", 'CALM': "C", 'South': "S", 'ESE': "E",
               'S': 'S',
               'Variable': "V", 'VAR': "V", 'N': "N", 'E': "E"}
    df['Wind_Direction'] = df['Wind_Direction'].map(mapWind)

    return df

def ColumnTransform(df):
    df['Month'] = pd.to_datetime(df["Weather_Timestamp"]).apply(lambda x: dt.datetime.date(x).month)
    df['Time'] = pd.to_datetime(df["Weather_Timestamp"]).apply(lambda x: dt.datetime.time(x).hour)
    df['Year'] = pd.to_datetime(df["Weather_Timestamp"]).apply(lambda x: dt.datetime.date(x).year)
    df['Duration'] = pd.to_datetime(df['End_Time']) - pd.to_datetime(df['Start_Time'])
    df['Duration'] = df['Duration'].apply(lambda x: round(x.total_seconds() / 60))
    df.drop(["Weather_Timestamp", "End_Time", "Start_Time"], axis=1, inplace=True)

    df = mappers(df)
    return df

def CleanNan(df):

    #clean variables with 20% of Nans
    df = df.dropna(subset=['Airport_Code', "Precipitation(in)", "Wind_Chill(F)", "City"])

    #Inputar valores
    toImp = ["Visibility(mi)", "Humidity(%)", "Pressure(in)"]
    for i in toImp:
        df[i] = df.groupby(["City", "Month", "Time"])[i].transform(lambda x: x.fillna(x.mean()))

    df = df.dropna(subset=['Visibility(mi)', "Humidity(%)", "Pressure(in)", "Wind_Direction", "Weather_Condition",
                           "Wind_Direction"])

    return df


def Sampling(df):

    seed = 234
    s1 = df[df["Severity"] == 1]
    s2 = df[df["Severity"] == 2]
    s3 = df[df["Severity"] == 3]
    s4 = df[df["Severity"] == 4]

    s2_downsample = resample(s2, replace=True, n_samples=len(s1), random_state=seed)
    s3_downsample = resample(s3, replace=True, n_samples=len(s1), random_state=seed)
    s4_downsample = resample(s4, replace=True, n_samples=len(s1), random_state=seed)
    df = pd.concat([s1, s2_downsample, s3_downsample, s4_downsample]).reset_index(drop=True)

    return df

def remove_outliers(df, name):
    # Percentil
    q25, q75 = np.percentile(df[name], 25), np.percentile(df[name], 75)
    # Interquartile range
    iqr = 1.5 * (q75 - q25)
    # Limites
    lower, upper = q25 - iqr, q75 + iqr
    # Remove the outliers
    df = df[(df[name] >= lower) & (df[name] <= upper)]
    return df

def CleanOutlier(df):
    df = remove_outliers(df, name='Temperature(F)')
    df = remove_outliers(df, name='Wind_Chill(F)')
    df = remove_outliers(df, name='Humidity(%)')
    df = remove_outliers(df, name='Pressure(in)')
    df = remove_outliers(df, name='Wind_Speed(mph)')
    return df

def RemoveColumns(df):
    df=df.drop(["Airport_Code", "Zipcode"], axis=1)
    return df
