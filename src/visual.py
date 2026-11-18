
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve


# 1. Compare actual values with predicted values
def plot_actual_vs_predicted(y_test, predictions, model_name):

    plt.figure(figsize=(8, 6))

    plt.scatter(y_test, predictions, alpha=0.5)

    # Perfect prediction line
    min_value = min(np.min(y_test), np.min(predictions))
    max_value = max(np.max(y_test), np.max(predictions))

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        label="Perfect prediction"
    )

    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# 2. Check overfitting
def plot_learning_curve(model, X, y, model_name):

    train_sizes, train_scores, validation_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=1
    )

    # Convert negative RMSE into positive RMSE
    train_rmse = -train_scores
    validation_rmse = -validation_scores

    # Calculate mean and standard deviation
    train_mean = np.mean(train_rmse, axis=1)
    train_std = np.std(train_rmse, axis=1)

    validation_mean = np.mean(validation_rmse, axis=1)
    validation_std = np.std(validation_rmse, axis=1)

    plt.figure(figsize=(9, 6))

    # Training curve
    plt.plot(
        train_sizes,
        train_mean,
        marker="o",
        label="Training RMSE"
    )

    # Validation curve
    plt.plot(
        train_sizes,
        validation_mean,
        marker="o",
        label="Validation RMSE"
    )

    # Standard deviation areas
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15
    )

    plt.fill_between(
        train_sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        alpha=0.15
    )

    plt.xlabel("Number of Training Samples")
    plt.ylabel("RMSE")
    plt.title(f"Learning Curve - {model_name}")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# 3. Compare training and test performance
def plot_train_test_comparison(
    y_train,
    y_test,
    train_predictions,
    test_predictions,
    model_name
):

    # Calculate metrics
    metrics = {
        "MAE": [
            mean_absolute_error(y_train, train_predictions),
            mean_absolute_error(y_test, test_predictions)
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(y_train, train_predictions)),
            np.sqrt(mean_squared_error(y_test, test_predictions))
        ],
        "R²": [
            r2_score(y_train, train_predictions),
            r2_score(y_test, test_predictions)
        ]
    }

    labels = ["Train", "Test"]
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (metric_name, values) in zip(axes, metrics.items()):

        bars = ax.bar(
            x,
            values,
            width,
            label=metric_name
        )

        ax.set_title(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.3)

        ax.bar_label(bars, fmt="%.3f", padding=3)

    fig.suptitle(f"Train vs Test Performance - {model_name}")
    plt.tight_layout()
    plt.show()
