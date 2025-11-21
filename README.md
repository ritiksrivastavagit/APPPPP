# NATA Multi-Mnt Ridge Predictor & Clustering (Demo)

This demo Streamlit app:
- Predicts multiple spending columns (MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds)
  from the inputs: Age, Income, Recency, Frequency, NumWebVisitsMonth, NumDealsPurchases, Kidhome, Teenhome, MaritalStatus.
- Assigns a KMeans cluster based on the same features.

Files:
- app.py
- pp.py
- model_reg.pkl
- model_cluster.pkl
- feature_columns.json
- requirements.txt

How to run:
1. pip install -r requirements.txt
2. streamlit run app.py

Replace the models with your real trained models for production.
