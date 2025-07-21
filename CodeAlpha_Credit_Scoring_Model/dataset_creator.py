# credit_dataset_creator.py

import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n = 1000

# Generate synthetic features
data = pd.DataFrame({
    "Income": np.random.normal(50000, 15000, n).astype(int),  # Annual income in USD
    "Debt": np.random.normal(15000, 5000, n).astype(int),     # Total debt in USD
    "Credit_History_Years": np.random.randint(1, 25, n),      # Credit history in years
    "Employment_Status": np.random.choice(
        ["Employed", "Unemployed", "Self-Employed", "Student"], n),
    "Loan_Amount": np.random.normal(10000, 3000, n).astype(int),  # Requested loan in USD
    "Age": np.random.randint(18, 70, n),                      # Age of applicant
    "Marital_Status": np.random.choice(
        ["Single", "Married", "Divorced", "Widowed"], n),
    "Credit_Score": np.random.randint(300, 850, n)            # Simulated credit score
})

# Define binary creditworthiness
data["Credit_Status"] = np.where(data["Credit_Score"] >= 600, "Good", "Bad")

# Drop credit score to simulate prediction target
data = data.drop(columns=["Credit_Score"])

print(data.head())

# Save to CSV
data.to_csv("credit_scoring_dataset.csv", index=False)

print("Dataset saved as 'credit_scoring_dataset.csv'")
