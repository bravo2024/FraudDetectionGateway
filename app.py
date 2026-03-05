import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fraud Detection AI Portfolio", page_icon="🧠", layout="wide")

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
st.title("🧠 Credit Card Fraud Detection Pipeline")
st.markdown("*An end-to-end Machine Learning portfolio project demonstrating Data Analysis, Model Comparison, and Live Inference.*")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Data Exploration (EDA)", "🤖 Model Comparison", "🛡️ Interactive Inference"])

# ==========================================
# TAB 1: DATA EXPLORATION
# ==========================================
with tab1:
    st.header("Understanding the Dataset")
    st.write("""
    This dataset contains real transactions made by European cardholders in September 2013. 
    To protect user privacy, the bank applied **PCA (Principal Component Analysis)** to hide identities, 
    leaving us with numerical features `V1` to `V28`, plus the `Time` and `Amount`.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("The Extreme Class Imbalance")
        st.write("In the real world, fraud is rare. This creates a massive challenge for Machine Learning models.")
        class_counts = df_full['Class'].value_counts().reset_index()
        class_counts.columns = ['Class', 'Count']
        class_counts['Label'] = class_counts['Class'].map({0: 'Legitimate (99.8%)', 1: 'Fraudulent (0.17%)'})
        
        fig_pie = px.pie(class_counts, values='Count', names='Label', 
                         color='Label', color_discrete_map={'Legitimate (99.8%)': '#00CC96', 'Fraudulent (0.17%)': '#EF553B'},
                         hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        st.subheader("Raw Data Sample")
        st.dataframe(df_full.head(10), height=300)
        
    st.info("💡 **How we solved the imbalance:** During training, we used **SMOTE (Synthetic Minority Over-sampling Technique)** to mathematically generate synthetic fraudulent transactions, teaching the AI what fraud looks like without biasing it toward legitimate swipes.")

# ==========================================
# TAB 2: MODEL COMPARISON
# ==========================================
with tab2:
    st.header("Evaluating Multiple Algorithms")
    st.write("We trained three different Machine Learning architectures on the SMOTE-balanced dataset. Here is how they performed on unseen test data.")
    
    # Format metrics into a DataFrame
    metrics_df = pd.DataFrame(metrics_dict).T.reset_index()
    metrics_df.rename(columns={'index': 'Model'}, inplace=True)
    
    # Highlight Recall since it's most important for fraud
    st.dataframe(metrics_df.style.highlight_max(subset=['Recall', 'F1_Score', 'Accuracy'], color='lightgreen', axis=0), use_container_width=True)
    
    st.write("### Why Recall Matters Most")
    st.write("In fraud detection, **Recall** is our most critical metric. It tells us: *Out of all the actual fraud that occurred, what percentage did our model successfully catch?* It is better to falsely flag a legitimate transaction (low precision) than to let a thief drain an account (low recall).")
    
    fig_bar = px.bar(metrics_df, x='Model', y=['Recall', 'Precision', 'F1_Score'], barmode='group',
                     title="Performance Metrics Comparison", text_auto='.2f')
    st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# TAB 3: INTERACTIVE INFERENCE
# ==========================================
with tab3:
    st.header("Live Inference Simulator")
    st.write("Instead of fake data, let's pull a **real historical transaction** from the dataset and see how our models evaluate it.")
    
    col_ctrl, col_result = st.columns([1, 2])
    
    with col_ctrl:
        selected_model_name = st.selectbox("Select ML Model to use:", list(models.keys()))
        transaction_type = st.radio("Pull a transaction that is actually:", ["Legitimate", "Fraudulent"])
        
        if st.button("🔄 Pull Random Transaction", type="primary"):
            # Filter df based on selection
            target_class = 1 if transaction_type == "Fraudulent" else 0
            subset = df_full[df_full['Class'] == target_class]
            
            # Pick a random row
            random_idx = np.random.randint(0, len(subset))
            selected_row = subset.iloc[random_idx:random_idx+1]
            
            st.session_state['selected_row'] = selected_row
            
    with col_result:
        if 'selected_row' in st.session_state:
            row = st.session_state['selected_row']
            true_class = row['Class'].values[0]
            true_label = "🚨 FRAUD" if true_class == 1 else "✅ LEGITIMATE"
            
            st.markdown(f"### True Status: **{true_label}**")
            st.write(f"**Transaction Amount:** ${row['Amount'].values[0]:.2f}")
            
            # Prepare data for model
            X_input = row.drop(['Class', 'Time'], axis=1, errors='ignore')
            
            # Ensure columns match training
            X_input = X_input[feature_names]
            
            # Scale
            X_scaled = scaler.transform(X_input)
            
            # Predict
            model = models[selected_model_name]
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]
            
            pred_label = "🚨 FRAUD DETECTED" if prediction == 1 else "✅ CLEARED"
            confidence = probabilities[1] if prediction == 1 else probabilities[0]
            
            st.markdown("---")
            st.markdown(f"### AI Prediction ({selected_model_name}):")
            if prediction == true_class:
                st.success(f"{pred_label} (Correct Match) - Confidence: {confidence*100:.2f}%")
            else:
                st.error(f"{pred_label} (Incorrect) - Confidence: {confidence*100:.2f}%")
                
            with st.expander("View Raw PCA Features"):
                st.dataframe(X_input.T)
        else:
            st.info("Click the button to pull a transaction and run the AI.")
