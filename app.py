import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import uuid
from datetime import datetime
import plotly.express as px
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fraud Detection Gateway", page_icon="🛡️", layout="wide")

# --- LOAD ML MODEL ---
@st.cache_resource
def load_ml_components():
    model_path = 'models/fraud_model.pkl'
    scaler_path = 'models/scaler.pkl'
    features_path = 'models/features.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(features_path)
        return model, scaler, feature_names
    return None, None, None

model, scaler, feature_names = load_ml_components()

# --- PREDICTION ENGINE ---
def predict_fraud(amount):
    """Uses the real Random Forest model if available, else falls back to mock."""
    if model is not None and scaler is not None and feature_names is not None:
        # The model expects 29 features (V1-V28, Amount)
        # In a real scenario, V1-V28 are PCA features from the transaction.
        # For this live simulation, we will generate random realistic PCA features
        # and append the actual amount.
        
        # Generate synthetic V1-V28 values (normally distributed around 0)
        v_features = np.random.normal(loc=0.0, scale=1.0, size=28).tolist()
        
        # Occasionally inject "fraud-like" outliers to trigger the model naturally
        if random.random() < 0.05: # 5% chance to inject heavy outliers
             v_features[3] = random.uniform(5.0, 10.0) # V4 is often high in fraud
             v_features[11] = random.uniform(-10.0, -5.0) # V12 is often low in fraud
             v_features[13] = random.uniform(-10.0, -5.0) # V14 is often low in fraud
             
        # Create input array
        input_data = v_features + [amount]
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Scale
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        is_fraud = bool(prediction == 1)
        confidence = probabilities[1] if is_fraud else probabilities[0]
        return is_fraud, confidence
        
    else:
        # Fallback Mock Logic
        is_fraud = amount > 4000
        confidence = random.uniform(0.85, 0.99) if is_fraud else random.uniform(0.90, 0.99)
        return is_fraud, confidence

# --- INITIALIZE STATE ---
if 'data' not in st.session_state:
    st.session_state.data = []
if 'running' not in st.session_state:
    st.session_state.running = False

# --- UI HEADER ---
st.title("🛡️ Real-Time Fraud Detection Gateway")
if model is not None:
    st.markdown("*🟢 Live ML Engine Active: Random Forest + SMOTE Pipeline*")
else:
    st.markdown("*🟡 Warning: Real ML model not found. Running in simulation mode.*")

col1, col2, col3, col4 = st.columns(4)
metric_total = col1.empty()
metric_fraud = col2.empty()
metric_legit = col3.empty()
metric_rate = col4.empty()

st.markdown("### 📊 Live Transaction Stream")

chart_col, table_col = st.columns([1, 2])
chart_placeholder = chart_col.empty()
table_placeholder = table_col.empty()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    if st.button("▶️ Start Live Stream" if not st.session_state.running else "⏹️ Stop Stream"):
        st.session_state.running = not st.session_state.running
    
    if st.button("🗑️ Clear Data"):
        st.session_state.data = []
        st.rerun()

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("This dashboard simulates a live feed of global transactions. The underlying ML Engine (Random Forest) evaluates each transaction in milliseconds to flag anomalies based on PCA features.")

# --- LIVE STREAM LOOP ---
if st.session_state.running:
    # Generate a new synthetic transaction
    amount = round(random.uniform(10.0, 5000.0), 2)
    device = random.choice(["mobile_ios", "mobile_android", "desktop", "unknown"])
    
    # Run through the REAL ML MODEL
    is_fraud, confidence = predict_fraud(amount)
    
    status = "🚨 FRAUD" if is_fraud else "✅ OK"
    
    new_txn = {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "ID": str(uuid.uuid4())[:8],
        "Amount ($)": amount,
        "Device": device,
        "Status": status,
        "Confidence": f"{confidence*100:.1f}%"
    }
    
    st.session_state.data.insert(0, new_txn)
    if len(st.session_state.data) > 50:
        st.session_state.data.pop()

# --- UPDATE DASHBOARD ---
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data)
    
    total_txns = len(df)
    fraud_txns = len(df[df["Status"] == "🚨 FRAUD"])
    legit_txns = total_txns - fraud_txns
    fraud_rate = (fraud_txns / total_txns) * 100 if total_txns > 0 else 0
    
    metric_total.metric("Total Transactions (Window)", total_txns)
    metric_fraud.metric("Fraud Blocked", fraud_txns, delta_color="inverse")
    metric_legit.metric("Legitimate", legit_txns)
    metric_rate.metric("Fraud Rate", f"{fraud_rate:.1f}%")
    
    def color_fraud(val):
        color = '#ff4b4b' if val == '🚨 FRAUD' else ''
        return f'background-color: {color}'
    
    styled_df = df.style.map(color_fraud, subset=['Status'])
    table_placeholder.dataframe(styled_df, use_container_width=True, height=400)
    
    status_counts = df["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    fig = px.pie(status_counts, values="Count", names="Status", 
                 color="Status", color_discrete_map={"✅ OK": "#00CC96", "🚨 FRAUD": "#EF553B"},
                 hole=0.4)
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
    chart_placeholder.plotly_chart(fig, use_container_width=True)

if st.session_state.running:
    time.sleep(1.0)
    st.rerun()
