from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def split_data(df):

    # Variable cible
    y = df["EU_Sales"]

    # Variables explicatives
    X = df.drop(columns=[
        "EU_Sales",
        "Global_Sales",
        "Rank"
    ])

    # Find the index of the highest EU_Sales value (fix overfitting)
    max_index = y.idxmax()

    # Keep the maximum value in the training set
    X_max = X.loc[[max_index]]
    y_max = y.loc[[max_index]]

    # Remove this observation before splitting
    X_remaining = X.drop(index=max_index)
    y_remaining = y.drop(index=max_index)

    # Séparation des données
    X_train, X_test, y_train, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.2,
        random_state=42
    )

    # Add the maximum observation to the training set
    X_train = pd.concat([X_train, X_max])
    y_train = pd.concat([y_train, y_max])

    print("Training data:", X_train.shape)
    print("Test data:", X_test.shape)

    return X_train, X_test, y_train, y_test


def create_preprocessor(X_train):

    # Identifier les types de colonnes
    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Encodage des variables catégorielles
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=10,
        sparse_output=False
    )

    # Combiner les transformations
    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    print("Numeric values:", numeric_features)
    print("Categorical values:", categorical_features)

    return preprocessor
