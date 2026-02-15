import pandas as pd
import numpy as np
import os

# Load the Kaggle CSV
df = pd.read_csv("synthetic_medical_triage.csv")

# 1. Map Categorical data
df['arrival_mode'] = df['arrival_mode'].map({'walk_in': 0, 'wheelchair': 1, 'ambulance': 2})

# 2. Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Shard the data for 3 hospitals
# By passing the DataFrame 'df' instead of 'df.values', Pandas handles the splitting
hospitals = np.array_split(df, 10) 

os.makedirs("data", exist_ok=True)

# Save the column names before splitting
column_names = df.columns

for i, hospital_data in enumerate(hospitals):
    # Explicitly convert to DataFrame and re-attach column names
    shard_df = pd.DataFrame(hospital_data, columns=column_names)
    
    shard_df.to_csv(f"data/hospital_{i}.csv", index=False)
    print(f"✅ Created hospital_{i}.csv with {len(shard_df)} records.")