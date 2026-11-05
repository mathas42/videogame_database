
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
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
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 20, None],
        "model__min_samples_leaf": [1, 2, 5],
        "model__max_features": [0.7, 1.0]
    }

    # Recherche des meilleurs paramètres
    rf_search = RandomizedSearchCV(
        rf_pipeline,
        param_distributions=rf_params,
        n_iter=10,
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
