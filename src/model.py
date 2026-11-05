
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline


def train_random_forest(preprocessor, X_train, y_train):

    # Création du pipeline
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            random_state=42,
            n_jobs=1
        ))
    ])

    # Hyperparamètres à tester
    rf_params = {
        "model__n_estimators": [200, 300, 500, 600, 700],
        "model__max_depth": [None, 10, 20, 30, 40, 50],
        "model__min_samples_split": [2, 5, 10, 15, 20],
        "model__min_samples_leaf": [1, 2, 4, 8, 10],
        "model__max_features": ["sqrt", 0.3, 0.5, 0.7, 1.0],
        "model__bootstrap": [True, False]
    }

    # Recherche des meilleurs paramètres
    rf_search = RandomizedSearchCV(
        rf_pipeline,
        param_distributions=rf_params,
        n_iter=50,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # Entraînement et optimisation
    rf_search.fit(X_train, y_train)

    # Résultats
    print("Best parameters RF:", rf_search.best_params_)
    print("RMSE CV RF:", -rf_search.best_score_)

    return rf_search

def train_hist_gradient_boosting(preprocessor, X_train, y_train):

    # Création du pipeline
    hgb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", HistGradientBoostingRegressor(random_state=42))
    ])

    # Hyperparamètres à tester
    hgb_params = {
        "model__max_iter": [100, 200, 300, 500, 1000],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15, 0.3],
        "model__max_leaf_nodes": [15, 30, 50, 75, 100],
        "model__max_depth": [None, 5, 10, 15, 20],
        "model__min_samples_leaf": [10, 20, 30, 40, 50],
        "model__l2_regularization": [0, 0.1, 1, 5, 10, 50]
    }

    # Recherche des meilleurs hyperparamètres
    hgb_search = RandomizedSearchCV(
        hgb_pipeline,
        param_distributions=hgb_params,
        n_iter=50,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # Entraînement et optimisation
    hgb_search.fit(X_train, y_train)

    # Affichage des résultats
    print("Meilleurs paramètres HGB :", hgb_search.best_params_)
    print("RMSE CV HGB :", -hgb_search.best_score_)

    return hgb_search
