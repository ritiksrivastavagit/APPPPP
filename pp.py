import json
import pandas as pd

with open("feature_columns.json","r") as f:
    FEATURE_COLUMNS = json.load(f)

MARITAL_CATEGORIES = ['Single', 'Married', 'Divorced', 'Together', 'Widow']

def preprocess_input(input_dict):
    """
    Build a single-row DataFrame with the correct FEATURE_COLUMNS order.
    Expected input keys:
      - Age, Income, Recency, Frequency, NumWebVisitsMonth, NumDealsPurchases, Kidhome, Teenhome
      - MaritalStatus (string)
    """
    row = {col: 0 for col in FEATURE_COLUMNS}

    # numeric fields
    for num in ['Age', 'Income', 'Recency', 'Frequency', 'NumWebVisitsMonth', 'NumDealsPurchases', 'Kidhome', 'Teenhome']:
        if num in input_dict and input_dict[num] is not None:
            try:
                row[num] = float(input_dict[num])
            except:
                row[num] = 0.0

    # Marital status -> one-hot
    ms = input_dict.get("MaritalStatus", None)
    if ms:
        key = "MaritalStatus_" + str(ms).strip().replace(" ", "_")
        if key in row:
            row[key] = 1

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
