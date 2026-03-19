import streamlit as st
import pandas as pd
import numpy as np
import joblib, shap, matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Predictor", page_icon="🔄", layout="wide")

@st.cache_resource
def load_artifacts():
    model  = joblib.load("model/best_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    feats  = open("model/feature_cols.txt").read().splitlines()
    return model, scaler, feats

model, scaler, FEATURES = load_artifacts()
NUM_FEATS = ["tenure","MonthlyCharges","TotalCharges","AvgMonthlySpend","ServiceCount"]

st.sidebar.header("Enter Customer Profile")
tenure   = st.sidebar.slider("Tenure (months)", 1, 72, 12)
monthly  = st.sidebar.slider("Monthly Charges ($)", 18, 120, 70)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month","One year","Two year"])
internet = st.sidebar.selectbox("Internet Service", ["Fiber optic","DSL","No"])
security = st.sidebar.toggle("Online Security", value=False)
support  = st.sidebar.toggle("Tech Support", value=False)
senior   = st.sidebar.toggle("Senior Citizen", value=False)
paper    = st.sidebar.toggle("Paperless Billing", value=True)

sample = pd.DataFrame([{c: 0 for c in FEATURES}])
raw = {
    "tenure": tenure, "MonthlyCharges": monthly,
    "TotalCharges": tenure*monthly, "AvgMonthlySpend": monthly,
    "SeniorCitizen": int(senior), "PaperlessBilling": int(paper),
    "ServiceCount": 2+int(security)+int(support),
    "NoSupportServices": int(not security and not support),
    "IsMonthToMonth": int(contract=="Month-to-month"),
    "TenureGroup": 0 if tenure<=12 else (1 if tenure<=36 else 2),
}
for k,v in raw.items():
    if k in sample.columns: sample[k]=v

for c in ["Contract_One year","Contract_Two year",
          "InternetService_Fiber optic","InternetService_No"]:
    if c in sample.columns: sample[c]=0

if contract=="One year" and "Contract_One year" in sample.columns:
    sample["Contract_One year"]=1
if contract=="Two year" and "Contract_Two year" in sample.columns:
    sample["Contract_Two year"]=1
if internet=="Fiber optic" and "InternetService_Fiber optic" in sample.columns:
    sample["InternetService_Fiber optic"]=1
if internet=="No" and "InternetService_No" in sample.columns:
    sample["InternetService_No"]=1

pres = [f for f in NUM_FEATS if f in sample.columns]
sample[pres] = scaler.transform(sample[pres])

prob = model.predict_proba(sample)[0][1]
risk = "HIGH RISK" if prob>=0.7 else ("MEDIUM RISK" if prob>=0.4 else "LOW RISK")

st.title("Customer Churn Predictor")
st.caption("Adjust the sliders on the left — predictions update instantly")

c1,c2,c3 = st.columns(3)
c1.metric("Churn Probability", f"{prob*100:.1f}%")
c2.metric("Risk Level", risk)
c3.metric("Recommendation", "Intervene now" if prob>=0.4 else "Low priority")

st.divider()
st.subheader("Why this prediction? (SHAP Explanation)")

try:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(sample)
    sv1 = sv[1][0] if isinstance(sv, list) else sv[0]
    top = pd.Series(sv1, index=FEATURES).abs().sort_values(ascending=False).head(10)
    colors = ["#ef4444" if sv1[FEATURES.index(f)]>0 else "#22c55e" for f in top.index]
    fig, ax = plt.subplots(figsize=(9,4))
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_xlabel("SHAP Impact")
    ax.set_title("Top factors driving this prediction")
    ax.spines[["top","right"]].set_visible(False)
    fig.patch.set_facecolor("white")
    st.pyplot(fig, use_container_width=True)
    st.caption("Red = pushing toward churn   |   Green = pushing toward retention")
except Exception as e:
    st.warning(f"SHAP explanation unavailable: {e}")
