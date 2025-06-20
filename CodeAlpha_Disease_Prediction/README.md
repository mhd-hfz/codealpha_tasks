# CodeAlpha_DiseasePrediction

## Overview
This project is part of the CodeAlpha Machine Learning Internship. It focuses on predicting the likelihood of diabetes in patients based on medical data from the PIMA Indians Diabetes dataset.

The goal is to apply various classification algorithms to structured medical data and evaluate their performance in disease prediction.

---

## Dataset
- **Name:** PIMA Indians Diabetes Dataset  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)  
- **Description:** The dataset contains medical diagnostic measurements such as glucose level, blood pressure, BMI, insulin level, age, and others for female patients of PIMA Indian heritage, along with a binary outcome indicating diabetes presence (1) or absence (0).

---

## Approach
- Data preprocessing including handling missing/zero values and feature scaling.
- Training multiple classification models:
  - Logistic Regression
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Decision Tree
  - Gradient Boosting Classifier
- Model evaluation using accuracy, confusion matrix, precision, recall, and F1-score.
- Feature importance analysis using Random Forest.

---

## Results Summary
- Models achieved approximately 70-75% accuracy on the test set.
- Logistic Regression and Random Forest performed comparably well.
- Feature importance visualization highlights the most influential features in predicting diabetes.

---

## How to Run
1. Clone this repository  
   ```bash
   git clone https://github.com/your-username/CodeAlpha_DiseasePrediction.git

2. Navigate into the project directory  
   ```bash
   cd CodeAlpha_DiseasePrediction

3. Install required Python libraries
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn

4. Launch Jupyter Notebook
   ```bash
   jupyter notebook main.ipynb

5. This will open Jupyter in your browser. If it doesn't open automatically, go to:
http://127.0.0.1:8888/tree

6. In the Jupyter interface, click on main.ipynb and run each cell to execute the project.

