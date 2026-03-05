import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fraud Detection AI Pipeline", page_icon="🛡️", layout="wide")

# --- LOAD DATA & MODELS ---
@st.cache_data
def load_sample_data():
    df = pd.read_csv('data/creditcard.csv')
    return df

@st.cache_resource
def load_ml_components():
    scaler = joblib.load('models/scaler.pkl')
    features = joblib.load('models/features.pkl')
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    models = {
        "Logistic Regression": joblib.load('models/logistic_regression.pkl'),
        "Decision Tree": joblib.load('models/decision_tree.pkl'),
        "Random Forest": joblib.load('models/random_forest.pkl')
    }
    return scaler, features, metrics, models

df_full = load_sample_data()
scaler, feature_names, metrics_dict, models = load_ml_components()

# --- UI HEADER ---
st.title("🛡️ Enterprise Fraud Detection Gateway")
st.markdown("""
*An end-to-end Machine Learning pipeline engineered to evaluate credit card transactions. 
Features Data Analysis, Model Benchmarking, and a Live Interactive Inference Engine with Business Threshold Tuning.*
""")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Data Architecture (EDA)", "🤖 Model Benchmarking", "🎚️ Live Inference & Threshold Tuning"])

# ==========================================
# TAB 1: DATA EXPLORATION
# ==========================================
with tab1:
    st.header("Understanding the Dataset Architecture")
    st.write("""
    This system processes anonymized credit card transactions. To comply with strict banking privacy regulations, 
    the original data underwent **PCA (Principal Component Analysis)**. This mathematical transformation hides personal 
    identities while preserving the variance and patterns necessary for Machine Learning. We are left with numerical 
    features `V1` to `V28`, plus `Time` and `Amount`.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("The Class Imbalance Problem")
        st.write("In enterprise environments, fraud is highly anomalous. Less than 0.2% of transactions in this dataset are fraudulent. This extreme imbalance will cause naive models to predict 'Legitimate' 100% of the time.")
        class_counts = df_full['Class'].value_counts().reset_index()
        class_counts.columns = ['Class', 'Count']
        class_counts['Label'] = class_counts['Class'].map({0: 'Legitimate (99.8%)', 1: 'Fraudulent (0.17%)'})
        
        fig_pie = px.pie(class_counts, values='Count', names='Label', 
                         color='Label', color_discrete_map={'Legitimate (99.8%)': '#00CC96', 'Fraudulent (0.17%)': '#EF553B'},
                         hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        st.subheader("Raw Data Vector Sample")
        st.dataframe(df_full.head(10), height=300)
        
    st.info("💡 **Engineering Solution:** To prevent the AI from becoming biased toward the majority class, I engineered a preprocessing pipeline utilizing **SMOTE (Synthetic Minority Over-sampling Technique)**. This dynamically synthesizes minority class vectors during training, establishing a balanced decision boundary.")

# ==========================================
# TAB 2: MODEL COMPARISON
# ==========================================
with tab2:
    st.header("Evaluating Algorithm Architectures")
    st.write("I trained and benchmarked three distinct algorithmic architectures on the SMOTE-balanced dataset. The results below reflect their performance on an unseen test holdout set.")
    
    metrics_df = pd.DataFrame(metrics_dict).T.reset_index()
    metrics_df.rename(columns={'index': 'Model'}, inplace=True)
    
    st.dataframe(metrics_df.style.highlight_max(subset=['Recall', 'F1_Score', 'Accuracy'], color='lightgreen', axis=0), use_container_width=True)
    
    st.write("### The Business Metric: Why Recall Overrules Accuracy")
    st.write("In banking security, **Recall** (Sensitivity) is the primary KPI. A model with 99.8% Accuracy is useless if it misses 50% of the fraud. We optimize to ensure we catch the maximum number of true fraudulent events, accepting a slight hit to Precision (false positives) to protect the business's bottom line.")
    
    fig_bar = px.bar(metrics_df, x='Model', y=['Recall', 'Precision', 'F1_Score'], barmode='group',
                     title="Performance Metrics Comparison", text_auto='.2f')
    st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# TAB 3: INTERACTIVE INFERENCE & THRESHOLD TUNING
# ==========================================
with tab3:
    st.header("Live Inference Engine & Business Threshold Tuner")
    st.write("""
    Machine Learning models output a *probability*, not a binary Yes/No. 
    By adjusting the **Decision Threshold**, we control the trade-off between customer friction (blocking innocent people) 
    and business risk (letting fraud slip through).
    """)
    
    col_ctrl, col_result = st.columns([1, 1.5])
    
    with col_ctrl:
        st.subheader("Engine Controls")
        selected_model_name = st.selectbox("1. Select AI Engine:", list(models.keys()))
        
        st.markdown("---")
        st.write("**2. Set Business Risk Threshold**")
        threshold = st.slider(
            "Fraud Probability Cutoff (%)", 
            min_value=1, max_value=99, value=50, step=1,
            help="If the AI's confidence is above this number, it blocks the transaction."
        )
        
        # Show business impact based on threshold
        if threshold < 30:
            st.warning("⚠️ **High Friction:** Catching all fraud, but blocking many innocent customers.")
        elif threshold > 80:
            st.error("🚨 **High Risk:** Very low customer friction, but sophisticated fraud will slip through.")
        else:
            st.success("⚖️ **Balanced:** Standard enterprise operating zone.")
            
        st.markdown("---")
        transaction_type = st.radio("3. Inject Historical Transaction Type:", ["Legitimate", "Fraudulent"])
        
        if st.button("🚀 Execute Inference", type="primary"):
            target_class = 1 if transaction_type == "Fraudulent" else 0
            subset = df_full[df_full['Class'] == target_class]
            random_idx = np.random.randint(0, len(subset))
            st.session_state['selected_row'] = subset.iloc[random_idx:random_idx+1]
            
    with col_result:
        if 'selected_row' in st.session_state:
            row = st.session_state['selected_row']
            true_class = row['Class'].values[0]
            true_label = "🚨 FRAUD" if true_class == 1 else "✅ LEGITIMATE"
            
            st.markdown("### 📡 Transaction Payload Intercepted")
            st.write(f"**True Ground Truth:** {true_label}")
            st.write(f"**Transaction Amount:** ${row['Amount'].values[0]:.2f}")
            
            # Prepare data
            X_input = row.drop(['Class', 'Time'], axis=1, errors='ignore')[feature_names]
            X_scaled = scaler.transform(X_input)
            
            # Predict Probabilities
            model = models[selected_model_name]
            probabilities = model.predict_proba(X_scaled)[0]
            
            fraud_prob = probabilities[1] * 100
            
            # Apply custom threshold
            is_blocked = fraud_prob >= threshold
            
            st.markdown("---")
            st.markdown("### 🧠 AI Evaluation & Business Logic")
            
            # Visual probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = fraud_prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"{selected_model_name} Fraud Probability"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, threshold], 'color': "lightgreen"},
                        {'range': [threshold, 100], 'color': "salmon"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold
                    }
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Result Banner
            if is_blocked:
                if true_class == 1:
                    st.success(f"🛡️ **ACTION: BLOCKED** (Correctly stopped fraud!)")
                else:
                    st.error(f"⚠️ **ACTION: BLOCKED** (False Positive! You just blocked an innocent customer.)")
            else:
                if true_class == 0:
                    st.success(f"✅ **ACTION: APPROVED** (Correctly allowed legitimate swipe.)")
                else:
                    st.error(f"🚨 **ACTION: APPROVED** (False Negative! A thief just stole the money!)")
                    
        else:
            st.info("Configure the engine and click 'Execute Inference' to begin.")
