import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def size_data(df):
    print("Number of line :", df.shape[0])
    print("Nombre of column :", df.shape[1])

def is_missing(df):
    return df.isnull().sum().sort_values(ascending=False)

def doublon(df):
    return df.duplicated().sum()

def describe(df):
    return df.describe(include="all").T

def repartition(df):
    
    # Définir les limites des 5 catégories
    limites = [
    -1,0.001, 0.01, 0.1, 1,
    20, float("inf")
    ]

    noms = [
    "<1 000",
    "1 000-10 000",
    "10 000-100 000 ",
    "100 000-1M",
    "1M-20M",
    "20M<"
    ]

    # Attribuer une catégorie à chaque jeu
    categories = pd.cut(
        df["EU_Sales"],
        bins=limites,
        labels=noms
    )

    return (categories.value_counts(normalize=True) * 100).sort_index()