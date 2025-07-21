# 📊 CodeAlpha_Credit_Scoring_Model

This project was developed as part of the CodeAlpha Machine Learning Internship. It demonstrates a complete machine learning pipeline to predict creditworthiness of applicants using synthetic financial data with classification models in Python.

---

## 🔍 Project Objective

Predict whether a loan applicant is a **Good** or **Bad** credit risk based on features such as:

- Income  
- Debt  
- Credit history length  
- Employment status  
- Loan amount  
- Age  
- Marital status  

The goal is to build and compare models that can classify credit status accurately and provide insights for credit risk assessment.

---

## 🧾 Dataset

The dataset used is synthetically generated (`credit_scoring_dataset.csv`) with 1000 samples and features including:

- `Income`: Annual income in USD  
- `Debt`: Total debt in USD  
- `Credit_History_Years`: Number of years of credit history  
- `Employment_Status`: Employment category  
- `Loan_Amount`: Loan requested in USD  
- `Age`: Applicant age  
- `Marital_Status`: Marital status  
- `Debt_to_Income`: Ratio of debt to income (engineered)  
- `Loan_to_Income`: Ratio of loan amount to income (engineered)  
- `Credit_Status`: Target variable (Good/Bad credit)

---

## 🛠️ Technologies Used

- Python 3.x  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  

---

## 🔬 Methodology

### 1. Data Generation & Feature Engineering  
- Created synthetic dataset using Python and NumPy.  
- Added features like Debt-to-Income and Loan-to-Income ratios to improve model input.

### 2. Data Preprocessing  
- Encoded categorical variables using Label Encoding.  
- Scaled numeric features with StandardScaler.  
- Split data into 80% training and 20% testing sets.

### 3. Model Training  
Trained and compared the following classifiers with balanced class weights:  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  

### 4. Evaluation Metrics  
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC AUC Score  

---

## ✅ Results Summary

| Model               | Accuracy | Precision | Recall | F1-score | ROC AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | ~0.50    | ~0.45     | ~0.50  | ~0.45    | ~0.54   |
| Decision Tree       | ~0.49    | ~0.46     | ~0.44  | ~0.45    | ~0.49   |
| Random Forest       | ~0.46    | ~0.44     | ~0.45  | ~0.44    | ~0.45   |
| XGBoost              | ~0.42    | ~0.42     | ~0.42  | ~0.42    | ~0.39   |

*Note: Results may vary due to random synthetic data.*

---

## 📝 Results Interpretation

- The models show modest predictive ability with accuracy around 42–50%, indicating the task is challenging due to synthetic data limitations and feature overlap between classes.  
- **Precision** reflects the proportion of correctly predicted good credit cases out of all predicted good credit. Moderate precision means some false positives (incorrectly predicting good credit) occur.  
- **Recall** indicates how many actual good credit cases the model successfully identifies. Moderate recall means some good applicants may be missed (false negatives).  
- **F1-score** balances precision and recall to provide a single performance measure.  
- **ROC AUC** above 0.5 shows the models perform better than random guessing, but still have room for improvement.  
- These results suggest the need for richer or real-world data, additional features, or advanced modeling techniques for better credit risk prediction.

---

## ▶️ How to Run

### 1. Clone or Download this Repository

```bash
git clone https://github.com/yourusername/CodeAlpha_Credit_Scoring_Model.git
cd CodeAlpha_Credit_Scoring_Model
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Dataset

```bash
python credit_dataset_creator.py
```

### 4. Train and Evaluate Models

```bash
python main.py
```

# Project Files Included

| File Name          | Description                              |
|--------------------|------------------------------------------|
| `dataset_creator.py` | 	Generates synthetic credit scoring dataset       |
| `credit_scoring_dataset.csv`  | The generated dataset used for model training     |
| `main.py`     | Main script to preprocess, train, and evaluate|
| `requirements.txt` | Python dependencies                        |
| `README.md`        | 	Project documentation                                 |

---