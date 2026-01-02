# 📌 Day 05 — Random Forest Customer Churn Prediction Pipeline

This project continues the 60-Day AIML series by introducing a **Random Forest** model for customer churn prediction using the Telco Customer Churn dataset. Unlike previous days, Day 05 focuses **only on Random Forest**—no comparisons are made yet. Comparisons and dashboards will be implemented in **Day 06**.

---

## 🎯 Objectives for Day 05

| Goal | Status |
|------|--------|
| Load & preprocess dataset | ✔ Done |
| Build Random Forest pipeline | ✔ Done |
| Hyperparameter tuning with GridSearchCV | ✔ Done |
| Model evaluation (Accuracy, Matrix, AUC) | ✔ Done |
| Export model `churn_rf_model.pkl` | ✔ Done |
| No comparison with KNN/LR today | 🚫 Moved to Day 06 |

---

## 📂 Project Structure

```plaintext
Day05_RandomForest_ChurnPipeline/
│
├── notebooks/
│   └── Day05_RF.ipynb
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # reused from Day 03
│
├── model/
│   └── churn_rf_model.pkl                      # exported model
│
└── README.md                                   # this file
```

---

## 🧠 Dataset Used

| Detail | Value |
|--------|--------|
| Name | Telco Customer Churn |
| Source | Kaggle |
| Target Column | `Churn` |
| Format | Classification (Yes/No → 1/0) |

The same dataset from **Day 03 & Day 04** is reused to ensure consistency when comparing models tomorrow in **Day 06**.

---

## ⚙️ Setup & Installation

### 1. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# or
source venv/bin/activate       # macOS / Linux
```

### 2. Install Dependencies
```bash
pip install scikit-learn pandas numpy seaborn matplotlib joblib
```

---

## 🚀 Run the Notebook

Open:
```
notebooks/Day05_RF.ipynnb
```

Execute each step to:
- Clean dataset  
- Build preprocessing pipeline  
- Train Random Forest  
- Evaluate metrics  
- Export model  

---

## 🧩 Modeling Approach

The model is built with a `Pipeline`:

| Component | Purpose |
|-----------|----------|
| StandardScaler | Scale numerical features |
| OneHotEncoder | Encode categorical features |
| RandomForestClassifier | Final ML model |

---

## 🛠 Hyperparameters Tuned

```python
param_grid = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [8, 12, 16],
    "rf__min_samples_split": [2, 5]
}
```

Optimized using:
```
GridSearchCV (cv=3, scoring="accuracy")
```

---

## 📊 Evaluation Metrics

| Metric | Purpose |
|--------|----------|
| Accuracy | Quick baseline indicator |
| Precision / Recall / F1 | Churn-business impact on false positives |
| Confusion Matrix | Misclassification analysis |
| ROC-AUC Score | Rank customer churn probability |

Example Expected Output:
```
Accuracy: ~0.80 to 0.83
ROC-AUC: ~0.84 to 0.87
```

---

## 💾 Model Export

```python
import os, joblib
os.makedirs("model", exist_ok=True)
joblib.dump(best_rf, "model/churn_rf_model.pkl")
```

Model saved to:
```
model/churn_rf_model.pkl
```

---

# 🔭 Next Step (Day 06 Preview)

| Day | Model | Focus |
|-----|--------|--------|
| Day 03 | KNN | Baseline model |
| Day 04 | Logistic Regression | Probability-based churn |
| **Day 05** | **Random Forest** | Higher accuracy, strongest candidate |
| **Day 06** | **Comparison + Dashboard** | Leaderboard & Model Picker UI |
---
