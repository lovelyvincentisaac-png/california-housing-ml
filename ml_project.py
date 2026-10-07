# -------------------------
# CALIFORNIA HOUSING ML PROJECT
# -------------------------

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------
# 1. LOAD AND PREPARE DATA
# -------------------------

def load_data():
    """Load the California Housing dataset and separate features and target."""

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing values: {df.isnull().sum().sum()}")

    return X, y


# -------------------------
# 2. EVALUATE A MODEL
# -------------------------

def evaluate_model(model_name, y_true, predictions):
    """Calculate and display regression performance metrics."""

    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    r2 = r2_score(y_true, predictions)

    print(f"\n{model_name} Performance:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    return mae, rmse, r2


# -------------------------
# 3. DISPLAY FEATURE IMPORTANCE
# -------------------------

def show_feature_importance(model, feature_names):
    """Display Random Forest feature importance."""

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nRandom Forest Feature Importance:")
    print(importance_df.to_string(index=False))


# -------------------------
# 4. VISUALISE FINAL MODEL
# -------------------------

def plot_predictions(y_true, predictions):
    """Plot actual house values against model predictions."""

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_true,
        predictions,
        alpha=0.4
    )

    min_value = min(y_true.min(), predictions.min())
    max_value = max(y_true.max(), predictions.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual Median House Value")
    plt.ylabel("Predicted Median House Value")
    plt.title("Random Forest: Actual vs Predicted House Values")

    plt.tight_layout()
    plt.show()


# -------------------------
# 5. SAVE FINAL MODEL
# -------------------------

def save_model(model):
    """Save the trained model so it can be used for inference."""

    os.makedirs("model", exist_ok=True)

    model_path = "model/random_forest_model.pkl"

    joblib.dump(model, model_path)

    print(f"\nTrained model saved to: {model_path}")


# -------------------------
# 6. MAIN TRAINING PIPELINE
# -------------------------

def main():

    # Load dataset
    X, y = load_data()

    # Create an 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # -------------------------
    # BASELINE: LINEAR REGRESSION
    # -------------------------

    linear_model = LinearRegression()

    linear_model.fit(X_train, y_train)

    linear_predictions = linear_model.predict(X_test)

    evaluate_model(
        "Linear Regression",
        y_test,
        linear_predictions
    )

    # -------------------------
    # FINAL MODEL: RANDOM FOREST
    # -------------------------

    random_forest = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    random_forest.fit(X_train, y_train)

    rf_predictions = random_forest.predict(X_test)

    evaluate_model(
        "Random Forest",
        y_test,
        rf_predictions
    )

    # -------------------------
    # CHECK FOR OVERFITTING
    # -------------------------

    train_predictions = random_forest.predict(X_train)

    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, rf_predictions)

    print("\nOverfitting Check:")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²:     {test_r2:.4f}")
    print(f"R² Gap:      {train_r2 - test_r2:.4f}")

    # -------------------------
    # CROSS-VALIDATION
    # -------------------------

    cv_scores = cross_val_score(
        random_forest,
        X_train,
        y_train,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    print("\n5-Fold Cross-Validation:")
    print(f"Fold scores: {cv_scores}")
    print(f"Mean R²:     {cv_scores.mean():.4f}")
    print(f"Std Dev:     {cv_scores.std():.4f}")

    # -------------------------
    # FEATURE IMPORTANCE
    # -------------------------

    show_feature_importance(
        random_forest,
        X.columns
    )

    # -------------------------
    # SAVE FINAL MODEL
    # -------------------------

    save_model(random_forest)

    # -------------------------
    # VISUALISE RESULTS
    # -------------------------

    plot_predictions(
        y_test,
        rf_predictions
    )


# -------------------------
# RUN PROJECT
# -------------------------

if __name__ == "__main__":
    main()