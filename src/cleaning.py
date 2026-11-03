import pandas as pd

def clean_data(df):
    df = df.dropna()
    df = df[df["EU_Sales"] != 0]
    return df.reset_index(drop=True)