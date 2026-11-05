import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_test, predictions, name):

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R²:", r2)

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2
    }