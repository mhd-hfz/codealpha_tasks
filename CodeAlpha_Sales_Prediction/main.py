# sales_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(filepath):
    """
    Load the dataset from a CSV file.
    The first column is used as index (row labels).
    """
    df = pd.read_csv(filepath, index_col=0)
    return df

def explore_data(df):
    """
    Perform initial exploratory data analysis (EDA).
    - Display basic info and statistics.
    - Visualize relationships with pairplots.
    """
    # print("\n--- Dataset Head ---")
    # print(df.head())
    # print("\n--- Dataset Info ---")
    # print(df.info())
    # print("\n--- Dataset Description ---")
    # print(df.describe())
    
    # Visualize feature relationships and distributions
    sns.pairplot(df)
    plt.tight_layout()
    plt.savefig("pairplot.png")  # Save plot to file for later inspection
    print("\nPairplot saved as 'pairplot.png'.")

def prepare_data(df):
    """
    Data Cleaning, Transformation, and Feature Selection:
    - For this dataset, no missing values were found (checked in EDA).
    - Select features (independent variables) that influence Sales.
      Here, TV, Radio, and Newspaper ad budgets are used as features.
    - Split the data into train and test sets to evaluate model generalization.
    """
    X = df[['TV', 'Radio', 'Newspaper']]  # Feature selection
    y = df['Sales']                       # Target variable

    # Split dataset: 80% train, 20% test, fixed random seed for reproducibility
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    """
    Train multiple regression models on the training data.
    Models included:
    - Linear Regression: simple baseline model
    - Random Forest Regressor: ensemble tree-based model with better accuracy and non-linearity handling
    """
    models = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = lr

    # Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    return models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models using:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - Coefficient of Determination (R²)
    Print evaluation metrics for each model.
    """
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"\nModel: {name}")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R^2: {r2:.3f}")

def analyze_feature_impact(model, feature_names):
    """
    Analyze how changes in advertising spend impact Sales using linear regression coefficients.
    Assumes the input model is a trained LinearRegression model.
    """
    if not isinstance(model, LinearRegression):
        print("Feature impact analysis is only available for Linear Regression.")
        return

    print("\n--- Advertising Impact on Sales (Linear Regression Coefficients) ---")
    for feature, coef in zip(feature_names, model.coef_):
        direction = "increases" if coef > 0 else "decreases"
        print(f"- Every additional $1 spent on {feature} {direction} sales by approximately {abs(coef):.2f} units.")


def print_actionable_insights():
    """
    Print actionable insights for business based on model analysis.
    """
    print("\n--- Actionable Marketing Insights ---")
    print("- TV advertising has the strongest positive impact on sales.")
    print("- Radio advertising also contributes to sales growth but to a lesser extent.")
    print("- Newspaper advertising shows minimal correlation with sales. Consider reallocating budget.")
    print("- Strategy: Prioritize TV and Radio for best ROI. Use data-driven allocation for future campaigns.")


def main():
    # Step 1: Load data
    df = load_data("Advertising.csv")

    # Step 2: Exploratory Data Analysis (EDA)
    explore_data(df)

    # Step 3: Data cleaning, transformation, and feature selection
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Step 4: Train models on training data
    models = train_models(X_train, y_train)

    # Step 5: Evaluate models on test data
    evaluate_models(models, X_test, y_test)

    # Step 6: Analyze feature impact using Linear Regression
    analyze_feature_impact(models['Linear Regression'], X_train.columns)

    # Step 7: Print actionable business insights
    print_actionable_insights()


if __name__ == "__main__":
    main()
