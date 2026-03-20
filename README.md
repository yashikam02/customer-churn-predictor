#  Customer Churn Predictor + SHAP Explainability

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow)](https://huggingface.co/spaces/yashikamann02/customer-churn-predictor)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20SHAP-orange)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

> Predicts which telecom customers will churn and explains WHY using SHAP explainability. Deployed as a live interactive web app.

🔗 **Live Demo:** [Click here to try the app](https://huggingface.co/spaces/yashikamann02/customer-churn-predictor)


---

## 📌 Problem Statement

Telecom companies lose 15–25% of customers annually to churn, costing billions in lost revenue. This project builds an end-to-end ML pipeline that:

- Predicts churn probability for any customer profile in real time
- Explains the top factors driving each prediction using SHAP
- Quantifies revenue at risk by customer segment
- Deployed as a live web app usable by non-technical teams
---

## 🎯 Results

| Model | ROC-AUC | F1 Score | CV AUC (5-fold) |
|---|---|---|---|
| **Random Forest** ⭐ | **0.8398** | **0.6349** | **0.8826** |
| Logistic Regression | 0.8360 | 0.6145 | 0.9027 |
| XGBoost | 0.8290 | 0.6030 | 0.9084 |

> Best Model: **Random Forest** (ROC-AUC = 0.8398)

---

## 🔍 Key Findings (from SHAP Analysis)

- 📋 **Contract type** is the #1 churn driver — month-to-month customers churn at **3× the rate** of annual plan holders
- 🔒 Customers **without OnlineSecurity + TechSupport** churn 45% more
- ⏱️ **First 12 months** is the highest-risk window for churn
- 💸 Fiber optic customers with **high monthly charges + no add-ons** are the most at-risk segment

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn, Random Forest |
| Explainability | SHAP |
| Class Imbalance | SMOTE |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit, Hugging Face Spaces |

---

## 📂 Project Structure
```
customer-churn-predictor/
├── app.py                                     # Streamlit web app
├── requirements.txt                           # Dependencies
├── Dockerfile                                 # Container config
├── Customer_Churn_Prediction_Colab.ipynb      # Full analysis notebook
└── model/
    ├── best_model.pkl                         # Trained Random Forest model
    ├── scaler.pkl                             # Fitted StandardScaler
    └── feature_cols.txt                       # Feature names
```

---

## 🗺️ Notebook Walkthrough

| Section | Description |
|---|---|
| 1 — Setup | Libraries, load IBM Telco dataset |
| 2 — EDA | Churn rates, distributions, heatmaps |
| 3 — Feature Engineering | 5 new features, encoding, SMOTE |
| 4 — Modelling | 3 models, ROC curves, threshold tuning |
| 5 — SHAP | Summary, waterfall, dependence plots |
| 6 — Widget | Interactive prediction with SHAP |
| 7 — Business Insights | Risk segmentation, revenue at risk |

---

## 🚀 Run Locally

```bash
git clone https://github.com/yashikamann02/customer-churn-predictor.git
cd customer-churn-predictor
pip install -r requirements.txt
streamlit run app.py
```


---

## 📊 Dataset

- **Source:** IBM Telco Customer Churn Dataset
- **Size:** 7,043 customers × 21 features
- **Target:** Churn (Yes/No) — 26.5% positive class

---

## 👤 Author

**Yashika Mann**
📧 yashikamann02@gmail.com
🔗 [LinkedIn]([https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/yashika-mann-64a50525a/))
🐙 [GitHub](https://github.com/yashikam02)

---
⭐ If you found this useful, consider starring the repo!
