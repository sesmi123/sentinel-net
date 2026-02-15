import streamlit as st
import torch
import numpy as np
import pandas as pd
import joblib
from task import TriageNet
from collections import OrderedDict

# 1. Load the "Translator" and Labels
enc = joblib.load("triage_encoder.joblib")
labels = joblib.load("label_mapping.joblib")

# 2. Load and Rebuild the Model
def load_model():
    # We need to know the input size. The encoder tells us this!
    input_dim = enc.get_feature_names_out().shape[0]
    num_classes = len(labels)
    
    model = TriageNet(input_dim, num_classes)
    
    # Load the .npz weights
    weights = np.load("final_model_weights.npz")
    params_dict = zip(model.state_dict().keys(), [torch.tensor(v) for v in weights.values()])
    state_dict = OrderedDict(params_dict)
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# 3. Streamlit UI
st.title("🏥 AI Medical Triage Assistant")
st.write("Enter patient vitals to predict urgency.")

# Create inputs for every column in your CSV (except the target)
# NOTE: These names must match your CSV column names exactly
age = st.number_input("Age", min_value=0, max_value=120, value=30)
hr = st.number_input("Heart Rate", value=80)
sbp = st.number_input("Systolic BP", value=120)
spo2 = st.slider("Oxygen Saturation (%)", 50, 100, 98)
temp = st.number_input("Body Temp (C)", value=37.0)
pain = st.slider("Pain Level (1-10)", 1, 10, 5)
chronic = st.number_input("Chronic Disease Count", 0, 10, 0)
er_visits = st.number_input("Previous ER Visits", 0, 20, 0)
arrival = st.selectbox("Arrival Mode", ["Walk-in", "Ambulance", "Public Transport"])

if st.button("Predict Triage Level"):
    # Create a DataFrame for the new patient
    new_data = pd.DataFrame([{
        'age': age, 'heart_rate': hr, 'systolic_blood_pressure': sbp,
        'oxygen_saturation': spo2, 'body_temperature': temp, 'pain_level': pain,
        'chronic_disease_count': chronic, 'previous_er_visits': er_visits,
        'arrival_mode': arrival
    }])
    
    # Transform using the saved encoder
    encoded_input = enc.transform(new_data)
    tensor_input = torch.tensor(encoded_input, dtype=torch.float32)
    
    # Predict!
    with torch.no_grad():
        output = model(tensor_input)
        prediction = torch.argmax(output, dim=1).item()
        result_label = labels[prediction]
        
    st.success(f"Recommended Triage: **{result_label}**")