
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


# 2. Analyze prediction errors (residuals)
def plot_residuals(y_test, predictions, model_name):

    residuals = np.asarray(y_test) - np.asarray(predictions)

    plt.figure(figsize=(8, 6))

    plt.scatter(predictions, residuals, alpha=0.5)

    # Zero error reference line
    plt.axhline(y=0, color="red", linestyle="--")

    plt.xlabel("Predicted Sales")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(f"Residual Plot - {model_name}")
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
