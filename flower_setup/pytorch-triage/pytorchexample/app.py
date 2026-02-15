import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
from collections import OrderedDict
import plotly.graph_objects as go
import plotly.express as px
from task import TriageNet

# Load the encoder and original labels
enc = joblib.load("triage_encoder.joblib")
original_labels = joblib.load("label_mapping.joblib")

# Define the mapping from model output (0,1,2,3) to triage levels
URGENCY_MAPPING = {
    0: "Low",
    1: "Medium", 
    2: "High",
    3: "Critical"
}

URGENCY_DESCRIPTIONS = {
    "Low": "Non-urgent cases, stable condition",
    "Medium": "Requires attention but not immediately life-threatening",
    "High": "Serious condition, rapid intervention required",
    "Critical": "Life-threatening condition, immediate care needed"
}

def load_model():
    input_dim = enc.get_feature_names_out().shape[0]
    num_classes = len(original_labels)
    
    model = TriageNet(input_dim, num_classes)
    
    try:
        weights = np.load("final_model_weights.npz")
        params_dict = zip(model.state_dict().keys(), [torch.tensor(v) for v in weights.values()])
        state_dict = OrderedDict(params_dict)
        
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        st.write("Expected keys:", list(model.state_dict().keys()))
        if 'weights' in locals():
            st.write("Loaded keys:", list(weights.keys()))
        raise

model = load_model()

# Feature importance calculation using gradient-based method
def calculate_feature_importance(model, input_tensor, feature_names):
    """Calculate feature importance using input gradients"""
    input_tensor.requires_grad = True
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    
    # Get gradient of predicted class w.r.t. input
    model.zero_grad()
    output[0, predicted_class].backward()
    
    # Importance = |gradient * input|
    importance = (input_tensor.grad.abs() * input_tensor.abs()).squeeze().detach().numpy()
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False).head(10)
    
    return importance_df

# Streamlit UI
st.set_page_config(page_title="AI Medical Triage", layout="wide")
st.title("🏥 AI Medical Triage Assistant")
st.write("Enter patient vitals to predict urgency with confidence scores and explainability.")

# Create two columns for layout
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Patient Information")
    
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    hr = st.number_input("Heart Rate (bpm)", value=80, min_value=30, max_value=200)
    sbp = st.number_input("Systolic BP (mmHg)", value=120, min_value=60, max_value=250)
    spo2 = st.slider("Oxygen Saturation (%)", 50, 100, 98)
    temp = st.number_input("Body Temp (°C)", value=37.0, min_value=34.0, max_value=42.0)
    pain = st.slider("Pain Level (1-10)", 1, 10, 5)
    chronic = st.number_input("Chronic Disease Count", 0, 10, 0)
    er_visits = st.number_input("Previous ER Visits", 0, 20, 0)
    arrival = st.selectbox("Arrival Mode", ["Walk-in", "Ambulance", "Public Transport"])
    
    predict_button = st.button("🔍 Predict Triage Level", use_container_width=True)

with col2:
    if predict_button:
        # Create DataFrame for the new patient
        new_data = pd.DataFrame([{
            'age': age, 'heart_rate': hr, 'systolic_blood_pressure': sbp,
            'oxygen_saturation': spo2, 'body_temperature': temp, 'pain_level': pain,
            'chronic_disease_count': chronic, 'previous_er_visits': er_visits,
            'arrival_mode': arrival
        }])
        
        # Transform and predict
        encoded_input = enc.transform(new_data)
        tensor_input = torch.tensor(encoded_input, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(tensor_input)
            probabilities = F.softmax(output, dim=1).squeeze()
            confidence_scores = probabilities.numpy()
            prediction = torch.argmax(output, dim=1).item()
            
            # Map numeric prediction to urgency level
            result_label = URGENCY_MAPPING[prediction]
            confidence = confidence_scores[prediction] * 100
        
        # Display prediction with confidence
        st.subheader("🎯 Triage Prediction")
        
        # Color coding based on urgency
        color_map = {
            "Critical": "🔴",
            "High": "🟠", 
            "Medium": "🟡",
            "Low": "🟢"
        }
        icon = color_map.get(result_label, "⚪")
        
        st.markdown(f"### {icon} **{result_label}** Priority")
        st.write(f"*{URGENCY_DESCRIPTIONS[result_label]}*")
        st.metric("Confidence Score", f"{confidence:.1f}%")
        
        # Confidence interpretation
        if confidence >= 80:
            st.success("✅ High confidence - Model is very certain about this prediction")
        elif confidence >= 60:
            st.warning("⚠️ Moderate confidence - Consider clinical review")
        else:
            st.error("❌ Low confidence - Manual assessment strongly recommended")
        
        # Confidence distribution chart - map all predictions to urgency levels
        st.subheader("📊 Confidence Distribution")
        urgency_labels = [URGENCY_MAPPING[i] for i in range(len(confidence_scores))]
        conf_df = pd.DataFrame({
            'Triage Level': urgency_labels,
            'Probability': confidence_scores * 100
        }).sort_values('Probability', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=conf_df['Probability'],
            y=conf_df['Triage Level'],
            orientation='h',
            marker=dict(
                color=conf_df['Probability'],
                colorscale='RdYlGn',
                showscale=False
            ),
            text=[f"{p:.1f}%" for p in conf_df['Probability']],
            textposition='auto',
        ))
        fig.update_layout(
            xaxis_title="Probability (%)",
            yaxis_title="Triage Level",
            height=300,
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance explanation
        st.subheader("🔍 Why This Prediction?")
        st.write("Top factors influencing this triage decision:")
        
        feature_names = enc.get_feature_names_out()
        importance_df = calculate_feature_importance(model, tensor_input.clone(), feature_names)
        
        # Normalize importance for visualization
        importance_df['Normalized'] = (importance_df['Importance'] / importance_df['Importance'].max()) * 100
        
        fig2 = go.Figure(go.Bar(
            x=importance_df['Normalized'],
            y=importance_df['Feature'],
            orientation='h',
            marker=dict(color='#FF6B6B'),
            text=[f"{v:.0f}%" for v in importance_df['Normalized']],
            textposition='auto',
        ))
        fig2.update_layout(
            xaxis_title="Relative Importance (%)",
            yaxis_title="",
            height=400,
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Clinical recommendations
        st.subheader("💡 Recommended Actions")
        recommendations = {
            "Critical": [
                "🚨 Immediate physician assessment required",
                "🛏️ Prepare ICU bed or critical care area",
                "💉 Start vital monitoring and IV access",
                "📞 Alert emergency response team"
            ],
            "High": [
                "⏱️ Assessment within 15 minutes",
                "🛏️ Assign to high-priority bed",
                "📊 Order stat labs and imaging as needed",
                "👨‍⚕️ Notify attending physician"
            ],
            "Medium": [
                "⏰ Assessment within 30-60 minutes",
                "🪑 Assign to standard waiting area",
                "📋 Complete intake documentation",
                "🩺 Monitor vital signs periodically"
            ],
            "Low": [
                "⏳ Assessment within 1-2 hours",
                "🪑 Standard waiting room assignment",
                "📝 Standard intake process",
                "ℹ️ Provide estimated wait time"
            ]
        }
        
        actions = recommendations.get(result_label, recommendations["Low"])
        for action in actions:
            st.write(action)
        
        # Safety notice
        st.info("⚕️ **Clinical Note**: This AI prediction is a decision support tool. Final triage decisions must be made by qualified healthcare professionals considering the complete clinical picture.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This AI triage system uses federated learning to predict patient urgency while maintaining privacy.
    
    **Triage Levels:**
    - 🔴 **Critical**: Life-threatening, immediate care needed
    - 🟠 **High**: Serious condition, rapid intervention required
    - 🟡 **Medium**: Requires attention but not immediately life-threatening
    - 🟢 **Low**: Non-urgent, stable condition
    
    **Confidence Scores:**
    - ≥80%: High confidence
    - 60-79%: Moderate confidence  
    - <60%: Low confidence (manual review)
    """)
    
    st.header("📈 Model Info")
    st.metric("Input Features", len(enc.get_feature_names_out()))
    st.metric("Triage Categories", len(original_labels))
    
    # Display the mapping
    st.write("**Urgency Level Mapping:**")
    for key, value in URGENCY_MAPPING.items():
        st.write(f"• {key} → {value}")