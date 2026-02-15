import pandas as pd
import numpy as np
import os

# Load the Kaggle CSV
df = pd.read_csv("synthetic_medical_triage.csv")

# 1. Normalize numerical features (Age, HR, O2, Temp)
# 2. Encode categorical (Arrival Mode: Ambulance vs Walk-in)
df['arrival_mode'] = df['arrival_mode'].map({'walk_in': 0, 'wheelchair': 1, 'ambulance': 2})

# 3. Shard the data for 3 hospitals
# np.array_split returns a list of arrays, so we convert each back to a DF
hospitals = np.array_split(df.sample(frac=1), 3)

os.makedirs("data", exist_ok=True)

for i, hospital_data in enumerate(hospitals):
    # Convert the numpy array shard back to a Pandas DataFrame
    shard_df = pd.DataFrame(hospital_data)
    shard_df.to_csv(f"data/hospital_{i}.csv", index=False)
    print(f"Created hospital_{i}.csv with {len(shard_df)} records.")