import streamlit as st
import joblib
import json
from pp import preprocess_input
import pandas as pd

st.set_page_config(page_title="NATA - Multi-Mnt Ridge Predictor & Clustering", layout="centered")
st.title("Customer Spend (multiple Mnt) Predictor + Clustering (Demo)")

# Load
reg_model = joblib.load("model_reg.pkl")
cluster_model = joblib.load("model_cluster.pkl")
with open("feature_columns.json","r") as f:
    feature_cols = json.load(f)

st.sidebar.header("Customer inputs (use realistic values)")
Age = st.sidebar.number_input("Age", 16, 100, 35)
Income = st.sidebar.number_input("Income", 0, 5_000_000, 45000)
Recency = st.sidebar.number_input("Recency (days since last purchase)", 0, 365, 30)
Frequency = st.sidebar.number_input("Frequency (number of purchases)", 0, 200, 12)
NumWebVisitsMonth = st.sidebar.number_input("NumWebVisitsMonth", 0, 50, 3)
NumDealsPurchases = st.sidebar.number_input("NumDealsPurchases", 0, 20, 2)
Kidhome = st.sidebar.number_input("Kidhome (count)", 0, 5, 0)
Teenhome = st.sidebar.number_input("Teenhome (count)", 0, 5, 0)
MaritalStatus = st.sidebar.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Together', 'Widow'])

input_dict = {
    "Age": Age, "Income": Income, "Recency": Recency, "Frequency": Frequency,
    "NumWebVisitsMonth": NumWebVisitsMonth, "NumDealsPurchases": NumDealsPurchases,
    "Kidhome": Kidhome, "Teenhome": Teenhome, "MaritalStatus": MaritalStatus
}

st.write("### Inputs")
st.write(input_dict)

if st.button("Predict all Mnt values + Cluster"):
    X = preprocess_input(input_dict)

    # Regression predictions (multi-target)
    preds = reg_model.predict(X)[0]
    target_names = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    results = pd.DataFrame({"Target": target_names, "Predicted": preds})
    results["Predicted"] = results["Predicted"].clip(lower=0).round(2)
    st.success("Predicted Mnt values")
    st.table(results)

    # Cluster assignment
    cluster_label = int(cluster_model.predict(X)[0])
    st.info(f"Assigned Cluster: {cluster_label}")

    # Show cluster center (if available)
    try:
        centers = cluster_model.cluster_centers_
        center = centers[cluster_label]
        center_df = pd.DataFrame([center], columns=feature_cols).T
        center_df.columns = ["ClusterCenter"]
        st.write("Cluster center (feature values):")
        st.dataframe(center_df)
    except Exception as e:
        st.write("Cluster centers unavailable:", e)

st.write("---")
st.write("Note: This demo is trained on synthetic data. Replace 'model_reg.pkl' and 'model_cluster.pkl' with models trained on your real data for production.")
