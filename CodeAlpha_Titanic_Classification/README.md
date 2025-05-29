# CodeAlpha Titanic Classification 🚢

This is a machine learning project developed during the CodeAlpha internship program.

## 🔍 Problem
Predict whether a passenger survived the Titanic disaster based on available demographic and travel information.

## 🧠 Features Used
- Passenger Class (Pclass)
- Sex (encoded as 0 for male, 1 for female)
- Age (missing values filled with median)
- Number of siblings/spouses aboard (SibSp)
- Number of parents/children aboard (Parch)
- Fare
- Embarkation port (one-hot encoded: C, Q, S)

## 🛠️ Model
- Random Forest Classifier with default parameters
- Accuracy: ~80%

## 📁 Files
- `Titanic_Classification.ipynb` — Jupyter Notebook with full code and analysis
- `titanic.csv` — dataset containing passenger information

## ▶️ How to Run
1. Make sure you have Python 3 and Jupyter Notebook installed.
2. Install the required libraries (if not already installed):

   ```bash
   pip install pandas matplotlib seaborn scikit-learn

3. Open the notebook:
   ```bash
   jupyter notebook Titanic_Classification.ipynb

4. This will open Jupyter in your browser. If it doesn't open automatically, go to:
http://127.0.0.1:8888/tree
5. In the Jupyter interface, click on Titanic_Classification.ipynb and run each cell to execute the project.

## ✅ Output
The model achieves approximately 80% accuracy using cleaned and engineered features, with feature importance visualization for interpretability.
