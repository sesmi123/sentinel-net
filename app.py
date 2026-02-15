# streamlit_triage_app.py

import streamlit as st
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="AI Triage System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .critical { background-color: #ffebee; border-left: 5px solid #c62828; }
    .high { background-color: #fff3e0; border-left: 5px solid #ef6c00; }
    .medium { background-color: #fff9c4; border-left: 5px solid #f9a825; }
    .low { background-color: #e8f5e9; border-left: 5px solid #2e7d32; }
</style>
""", unsafe_allow_html=True)

# Define Neural Network Architecture (same as training)
class TriageNet(nn.Module):
    def __init__(self, input_dim=9, hidden_dims=[128, 64, 32, 16], num_classes=4, dropout_rates=[0.3, 0.3, 0.2, 0.2]):
        super(TriageNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.dropout1 = nn.Dropout(dropout_rates[0])
        
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout2 = nn.Dropout(dropout_rates[1])
        
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.dropout3 = nn.Dropout(dropout_rates[2])
        
        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.dropout4 = nn.Dropout(dropout_rates[3])
        
        self.fc5 = nn.Linear(hidden_dims[3], num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout4(x)
        
        x = self.fc5(x)
        return x

# Cache model loading
@st.cache_resource
def load_model():
    """Load model, scaler, and metadata"""
    device = torch.device('cpu')
    
    # Load model
    model = TriageNet(input_dim=9, num_classes=4)
    model.load_state_dict(torch.load('best_triage_model.pth', map_location=device))
    model.eval()
    
    # Load scaler and metadata
    scaler = joblib.load('scaler_20260215_044841.pkl')
    feature_names = joblib.load('feature_names_20260215_044841.pkl')
    metadata = joblib.load('model_metadata_20260215_044841.pkl')
    
    return model, scaler, feature_names, metadata, device

# Prediction function
def predict_triage(patient_data, model, scaler, device):
    """Make prediction for a patient"""
    # Convert to DataFrame
    feature_order = [
        'age', 'heart_rate', 'systolic_blood_pressure', 
        'oxygen_saturation', 'body_temperature', 'pain_level',
        'chronic_disease_count', 'previous_er_visits', 'respiratory_effort'
    ]
    
    patient_df = pd.DataFrame([patient_data])[feature_order]
    
    # Scale
    patient_scaled = scaler.transform(patient_df)
    patient_tensor = torch.FloatTensor(patient_scaled).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(patient_tensor)
        proba = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(output, dim=1).item()
    
    probabilities = proba.cpu().numpy()
    
    return pred, probabilities

# Load model
try:
    model, scaler, feature_names, metadata, device = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.info("Please ensure these files are in the same directory as this app:")
    st.code("""
    - best_triage_model.pth
    - scaler_20260215_044841.pkl
    - feature_names_20260215_044841.pkl
    - model_metadata_20260215_044841.pkl
    """)
    model_loaded = False

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 AI-Powered Emergency Triage System</h1>', unsafe_allow_html=True)
    
    if not model_loaded:
        return
    
    # Sidebar - Model Info
    with st.sidebar:
        st.header("📊 Model Information")
        st.metric("Model Accuracy", f"{metadata['test_accuracy']*100:.2f}%")
        st.metric("Total Training Samples", f"{metadata['train_samples']:,}")
        st.metric("Architecture", metadata['architecture'])
        
        st.markdown("---")
        st.subheader("🎯 Triage Levels")
        st.markdown("""
        - **Level 0 (Low)**: Stable, non-urgent
        - **Level 1 (Medium)**: Requires attention
        - **Level 2 (High)**: Serious, rapid intervention
        - **Level 3 (Critical)**: Life-threatening, immediate care
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🩺 Patient Assessment", "📈 Model Performance", "ℹ️ About"])
    
    with tab1:
        st.header("Enter Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=50, step=1)
            
            st.subheader("Medical History")
            chronic_disease_count = st.number_input("Chronic Disease Count", min_value=0, max_value=10, value=0, step=1)
            previous_er_visits = st.number_input("Previous ER Visits", min_value=0, max_value=20, value=0, step=1)
        
        with col2:
            st.subheader("Vital Signs")
            heart_rate = st.slider("Heart Rate (bpm)", min_value=30, max_value=200, value=80, step=1)
            systolic_blood_pressure = st.slider("Systolic BP (mmHg)", min_value=60, max_value=220, value=120, step=1)
            oxygen_saturation = st.slider("Oxygen Saturation (%)", min_value=70.0, max_value=100.0, value=98.0, step=0.1)
            body_temperature = st.slider("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)
        
        with col3:
            st.subheader("Clinical Assessment")
            pain_level = st.slider("Pain Level (1-10)", min_value=0, max_value=10, value=3, step=1)
            respiratory_effort = st.slider("Respiratory Effort", min_value=0, max_value=5, value=0, step=1,
                                          help="0 = Normal, 5 = Severe distress")
        
        # Predict button
        st.markdown("---")
        if st.button("🔍 Predict Triage Level", use_container_width=True):
            # Prepare patient data
            patient_data = {
                'age': age,
                'heart_rate': heart_rate,
                'systolic_blood_pressure': systolic_blood_pressure,
                'oxygen_saturation': oxygen_saturation,
                'body_temperature': body_temperature,
                'pain_level': pain_level,
                'chronic_disease_count': chronic_disease_count,
                'previous_er_visits': previous_er_visits,
                'respiratory_effort': respiratory_effort
            }
            
            # Make prediction
            pred_level, probabilities = predict_triage(patient_data, model, scaler, device)
            
            # Display results
            st.markdown("---")
            st.header("🎯 Prediction Results")
            
            triage_labels = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}
            triage_colors = {0: 'low', 1: 'medium', 2: 'high', 3: 'critical'}
            triage_emojis = {0: '✅', 1: '⚠️', 2: '🚨', 3: '🆘'}
            
            # Main prediction
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="result-box {triage_colors[pred_level]}">
                    <h2>{triage_emojis[pred_level]} Triage Level: {pred_level} ({triage_labels[pred_level]})</h2>
                    <h3>Confidence: {probabilities[pred_level]*100:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendations
                if pred_level == 3:
                    st.error("⚠️ **CRITICAL**: Immediate medical attention required. Alert emergency team.")
                elif pred_level == 2:
                    st.warning("🚨 **HIGH PRIORITY**: Rapid assessment and intervention needed.")
                elif pred_level == 1:
                    st.info("⚠️ **MEDIUM PRIORITY**: Patient requires attention but not immediately life-threatening.")
                else:
                    st.success("✅ **LOW PRIORITY**: Stable condition. Can wait for standard care.")
            
            with col2:
                # Probability gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probabilities[pred_level] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "gray"},
                            {'range': [50, 75], 'color': "lightblue"},
                            {'range': [75, 100], 'color': "blue"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Probability distribution
            st.subheader("📊 Full Probability Distribution")
            
            prob_df = pd.DataFrame({
                'Triage Level': [f"Level {i}: {triage_labels[i]}" for i in range(4)],
                'Probability': probabilities * 100
            })
            
            fig_bar = px.bar(
                prob_df,
                x='Triage Level',
                y='Probability',
                color='Probability',
                color_continuous_scale=['green', 'yellow', 'orange', 'red'],
                text='Probability'
            )
            fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_bar.update_layout(
                yaxis_title="Probability (%)",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Patient summary
            with st.expander("📋 View Patient Summary"):
                summary_df = pd.DataFrame([patient_data]).T
                summary_df.columns = ['Value']
                st.dataframe(summary_df, use_container_width=True)
    
    with tab2:
        st.header("📈 Model Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Test Accuracy", f"{metadata['test_accuracy']*100:.2f}%")
        col2.metric("Weighted F1-Score", f"{metadata['test_weighted_f1']:.4f}")
        col3.metric("Macro F1-Score", f"{metadata['test_macro_f1']:.4f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training History")
            st.info(f"""
            - **Total Epochs**: {metadata['total_epochs']}
            - **Best Validation Loss**: {metadata['best_val_loss']:.4f}
            - **Training Samples**: {metadata['train_samples']:,}
            - **Validation Samples**: {metadata['val_samples']:,}
            - **Test Samples**: {metadata['test_samples']:,}
            """)
        
        with col2:
            st.subheader("Model Architecture")
            st.code(f"""
Input Features: {metadata['num_features']}
Architecture: {metadata['architecture']}
Output Classes: {metadata['num_classes']}
            """)
        
        st.markdown("---")
        st.subheader("🎯 Features Used")
        features_df = pd.DataFrame(metadata['input_features'], columns=['Feature Name'])
        features_df.index = features_df.index + 1
        st.dataframe(features_df, use_container_width=True)
    
    with tab3:
        st.header("ℹ️ About This System")
        
        st.markdown("""
        ### AI-Powered Emergency Triage System
        
        This system uses a deep neural network to predict emergency triage priority levels based on patient vital signs and medical history.
        
        #### 🎯 How It Works
        1. **Input Patient Data**: Enter vital signs, medical history, and clinical assessments
        2. **AI Analysis**: The neural network processes 9 key features
        3. **Triage Prediction**: Get instant triage level with confidence scores
        4. **Clinical Decision Support**: Receive recommendations for patient management
        
        #### 🧠 Model Details
        - **Architecture**: Deep Neural Network (128-64-32-16 neurons)
        - **Training Data**: ~18,000 synthetic patient records
        - **Features**: 9 clinical and demographic variables
        - **Output**: 4 triage levels (Low, Medium, High, Critical)
        
        #### ⚠️ Important Notes
        - This system is designed as a **decision support tool**
        - Should be used **in conjunction with clinical judgment**
        - Not a replacement for medical professionals
        - Always follow your institution's triage protocols
        
        #### 🔒 Privacy & Security
        - No patient data is stored
        - All predictions are performed locally
        - Complies with healthcare data privacy standards
        
        ---
        
        **Developed for**: InnovAIte Hackathon - Pandemic Preparedness Challenge  
        **Model Version**: {metadata['timestamp']}
        """)

if __name__ == "__main__":
    main()