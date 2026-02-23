import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    return pd.read_csv(path)

def scale_data(df):
    df2 = df.drop("country", axis=1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df2)

    columns = df2.columns
    return pd.DataFrame(scaled, columns=columns)
