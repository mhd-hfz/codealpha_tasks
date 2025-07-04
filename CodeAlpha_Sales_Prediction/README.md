# 📊 CodeAlpha_Sales_Prediction

This project was developed as part of the CodeAlpha Data Science Internship. It demonstrates a complete machine learning pipeline for predicting product sales based on advertising spend using regression models in Python.

---

## 🔍 Project Objective

Predict future sales based on advertising spend across three media channels:
- **TV**
- **Radio**
- **Newspaper**

We aim to:
- Understand how each channel influences sales.
- Train machine learning models to predict sales.
- Provide actionable insights for marketing strategy.

---

## 🧾 Dataset

The dataset used is `Advertising.csv`, which contains 200 observations and 4 columns:
- `TV`: Advertising spend on TV (in $ thousands)
- `Radio`: Advertising spend on Radio (in $ thousands)
- `Newspaper`: Advertising spend on Newspaper (in $ thousands)
- `Sales`: Product sales (in $ millions)

---

## 🛠️ Technologies Used

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## 🔬 Methodology

### 1. Data Exploration
- Used Seaborn to create a pairplot and explore relationships between features.
- Summary statistics and distribution analysis.

### 2. Data Preparation
- Features selected: `TV`, `Radio`, `Newspaper`
- Target variable: `Sales`
- Train-test split: 80% training / 20% testing

### 3. Model Training
- **Linear Regression**
- **Random Forest Regressor**

### 4. Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score (Coefficient of Determination)

### 5. Business Insights
- Used Linear Regression coefficients to interpret impact of each advertising channel.
- Provided actionable recommendations based on results.

---

## ✅ Results

| Model              | MAE   | RMSE  | R² Score |
|-------------------|-------|-------|----------|
| Linear Regression | 1.46  | 1.78  | 0.899    |
| Random Forest     | 0.62  | 0.77  | 0.981    |

- **Random Forest performed better overall.**
- **TV and Radio spend have the most significant positive impact on sales.**
- **Newspaper ads have little to no effect.**

---

## 📌 Insights

- Increasing **TV** and **Radio** advertising budgets leads to higher predicted sales.
- **Newspaper** spending shows minimal return and may not be cost-effective.
- Focus on channels with proven ROI using data-driven decisions.

---

## ▶️ How to Run

### 1. Clone or Download this Repository

```bash
git clone https://github.com/yourusername/CodeAlpha_Sales_Prediction.git
cd CodeAlpha_Sales_Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Script

```bash
python sales_prediction.py
```

### 4. Jupyter Users (Optional)
If you're using Jupyter Notebook:
1. Launch Jupyter with:

```bash
jupyter notebook
```

2. This will open Jupyter in your browser. If it doesn't open automatically, go to:
http://127.0.0.1:8888/tree

3. Open and run sales_prediction.py or convert to .ipynb if needed.


# Project Files Included

| File Name          | Description                              |
|--------------------|------------------------------------------|
| `main.py` | Main Python script for the project       |
| `Advertising.csv`  | Dataset used for training and testing     |
| `pairplot.png`     | EDA visualization of feature relationships|
| `requirements.txt` | Python dependencies                        |
| `README.md`        | This file                                 |

---