import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib

def load_data(file_path="synthetic_medical_triage.csv"):
    df = pd.read_csv(file_path)
    
    # 1. DEBUG: See what the columns actually are
    print("Columns found in CSV:", df.columns.tolist())
    
    # Target column name - adjust this if the print statement above shows something different!
    target_col = "triage_classification.urgency_category"
    
    if target_col not in df.columns:
        # Fallback: try to find a column that contains 'urgency' or 'triage'
        potential_targets = [c for c in df.columns if 'urgency' in c.lower() or 'triage' in c.lower()]
        if potential_targets:
            target_col = potential_targets[0]
            print(f"Target not found. Switching to: {target_col}")
        else:
            raise KeyError(f"Could not find target column. Available: {df.columns.tolist()}")

    # 2. Feature/Target Separation
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3. Robust Encoding
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_enc = enc.fit_transform(X)

    # 4. Numeric Target Mapping
    labels, y_enc = np.unique(y, return_inverse=True)

    # 5. Stratified Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    joblib.dump(enc, "triage_encoder.joblib")
    joblib.dump(labels, "label_mapping.joblib")
    
    # 6. Return 4 items to match your client.py
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )