# 🔄 Customer Churn Prediction + SHAP Explainability

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](streamlittobeadded)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest-orange)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

> **Predicts which telecom customers will churn — and explains WHY
> using SHAP explainability. Deployed as a live interactive web app.**

🔗 **Live Demo:** [Click here to try the app](streamlittobeadded)

---

## 📌 Problem Statement

Telecom companies lose **15–25% of customers annually** to churn,
costing billions in lost revenue. The challenge isn't just predicting
who will leave — it's understanding *why*, so retention teams can act.

This project builds an end-to-end ML pipeline that:
- Predicts churn probability for any customer profile
- Explains the top factors driving each prediction (SHAP)
- Quantifies revenue at risk by customer segment
- Deploys as a live web app usable by non-technical teams

---

## 🎯 Results

| Model | ROC-AUC | F1 Score | CV AUC (5-fold) |
|---|---|---|---|
| **XGBoost** ⭐ | **0.853** | **0.631** | **0.847** |
| Random Forest | 0.841 | 0.614 | 0.836 |
| Logistic Regression | 0.827 | 0.598 | 0.821 |

> Model selected: **XGBoost** with optimal decision threshold tuning

---

## 🔍 Key Findings (from SHAP Analysis)

- 📋 **Contract type** is the #1 churn driver — month-to-month customers
  churn at **3× the rate** of annual plan holders
- 🔒 Customers **without OnlineSecurity + TechSupport** churn 45% more
- ⏱️ **First 12 months** is the highest-risk window for churn
- 💸 Fiber optic customers with **high monthly charges + no add-ons**
  are the most at-risk segment
- 💰 Estimated **$XX,XXX monthly revenue at risk** from high-risk segment

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP (SHapley Additive exPlanations) |
| Class Imbalance | SMOTE (imbalanced-learn) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Deployment | Streamlit, Streamlit Community Cloud |

---

## 📂 Project Structure

```
customer-churn-predictor/
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── Customer_Churn_Prediction_Colab.ipynb  # Full analysis notebook
└── model/
    ├── best_model.pkl        # Trained XGBoost model
    ├── scaler.pkl            # Fitted StandardScaler
    └── feature_cols.txt      # Feature names for inference
```

---

## 🗺️ Notebook Walkthrough

| Section | Description |
|---|---|
| 1 — Setup | Install libraries, load IBM Telco dataset |
| 2 — EDA | Churn rates, distributions, categorical heatmap |
| 3 — Feature Engineering | 5 new features, encoding, SMOTE |
| 4 — Modelling | 3 models, ROC curves, threshold optimisation |
| 5 — SHAP | Summary, waterfall, dependence plots |
| 6 — Widget | Interactive prediction with live SHAP explanation |
| 7 — Business Insights | Risk segmentation, revenue at risk, recommendations |

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/customer-churn-predictor.git
cd customer-churn-predictor

# Install dependencies
pip install -r requirements.txt

# Launch the app
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
