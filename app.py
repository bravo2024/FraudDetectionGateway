import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fraud Detection AI Pipeline", page_icon="🎓", layout="wide")

# --- LOAD DATA & MODELS ---
@st.cache_data
def load_sample_data():
    df = pd.read_csv('data/creditcard.csv')
    return df

@st.cache_data
def compute_correlations(df):
    corr = df.corr()['Class'].drop('Class').sort_values()
    return corr

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

@st.cache_data
def generate_evaluation_data(_df_full, _scaler, _features, _models):
    X = _df_full.drop(['Class', 'Time'], axis=1, errors='ignore')[_features]
    y = _df_full['Class'].astype(int)
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_scaled = _scaler.transform(X_test)
    
    roc_data = {}
    cm_data = {}
    
    for name, model in _models.items():
        y_probs = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        
        y_pred = model.predict(X_test_scaled)
        cm_data[name] = confusion_matrix(y_test, y_pred)
        
    return roc_data, cm_data, y_test

@st.cache_data
def get_batch_data(df):
    # To prevent browser crash, we take all 492 frauds + 9508 random legitimate = 10,000 rows
    frauds = df[df['Class'] == 1]
    legits = df[df['Class'] == 0].sample(n=9508, random_state=42)
    batch = pd.concat([frauds, legits]).sample(frac=1, random_state=42).reset_index(drop=True)
    return batch

df_full = load_sample_data()
correlations = compute_correlations(df_full)
scaler, feature_names, metrics_dict, models = load_ml_components()
roc_data, cm_data, y_test_true = generate_evaluation_data(df_full, scaler, feature_names, models)
batch_df = get_batch_data(df_full)

# --- UI HEADER ---
st.title("🎓 Machine Learning Concepts: Fraud Detection")
st.markdown("""
*An academic and technical demonstration of Machine Learning concepts, engineered by Vivek (@bravo2024).*  
*Designed to showcase the mathematical theory, loss functions, and dataset classification results.*
""")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Data & Mathematical Theory", "🧠 Algorithms & Evaluation", "🗄️ Batch Inference & Error Analysis"])

# ==========================================
# TAB 1: DATA & MATHEMATICAL ANALYSIS
# ==========================================
with tab1:
    st.header("1. The Dataset: Origin & Context")
    st.write("""
    **Source:** The official European Credit Card Fraud dataset.  
    **Volume:** 284,807 total transactions.  
    **The Problem:** We are looking for **492** fraudulent transactions hidden among **284,315** legitimate ones.
    """)
    
    st.markdown("---")
    st.header("2. The Mathematical Foundations")
    
    col_math1, col_math2 = st.columns(2)
    
    with col_math1:
        st.subheader("Data Privacy via PCA")
        st.write("Banks use **Principal Component Analysis (PCA)** to hide user identities while preserving statistical variance.")
        st.latex(r"Z = XW")
        st.write("Where $X$ is the original data, $W$ is the eigenvector matrix, and $Z$ represents the new orthogonal features ($V1$ to $V28$).")
        
    with col_math2:
        st.subheader("Solving Imbalance via SMOTE")
        st.write("To prevent the model from blindly guessing 'Legitimate' 100% of the time, I used **SMOTE** to algorithmically generate synthetic fraud vectors.")
        st.latex(r"x_{new} = x_i + \lambda (x_{zi} - x_i)")
        st.write("It finds a fraudulent point $x_i$, looks at its $k$-nearest neighbors, and interpolates a new point $x_{new}$.")
        
    st.markdown("---")
    st.header("3. Feature Correlation Analysis")
    st.write("Before training, a Data Scientist must understand which features the model will find most important.")
    
    col_corr1, col_corr2 = st.columns([1, 1.5])
    
    with col_corr1:
        st.subheader("Correlation Matrix")
        top_neg = correlations.head(5)
        top_pos = correlations.tail(5)
        top_corr = pd.concat([top_neg, top_pos]).reset_index()
        top_corr.columns = ['Feature', 'Correlation']
        
        fig_corr = px.bar(top_corr, x='Correlation', y='Feature', orientation='h', color='Correlation', color_continuous_scale='RdBu')
        fig_corr.update_layout(height=350)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with col_corr2:
        st.subheader("Cluster Separation (V14 vs V4)")
        fraud_points = df_full[df_full['Class'] == 1]
        legit_points = df_full[df_full['Class'] == 0].sample(n=2000, random_state=42)
        scatter_data = pd.concat([fraud_points, legit_points])
        scatter_data['Label'] = scatter_data['Class'].map({0: 'Legitimate', 1: 'Fraudulent'})
        
        fig_scatter = px.scatter(scatter_data, x='V14', y='V4', color='Label', color_discrete_map={'Legitimate': '#00CC96', 'Fraudulent': '#EF553B'}, opacity=0.6)
        fig_scatter.update_layout(height=350)
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# TAB 2: ALGORITHMS & EVALUATION
# ==========================================
with tab2:
    st.header("1. The Algorithms & Their Loss Functions")
    st.write("To demonstrate mastery, I trained three different types of algorithms. Here is how they mathematically 'learn' from the data:")
    
    col_alg1, col_alg2, col_alg3 = st.columns(3)
    
    with col_alg1:
        st.info("**Logistic Regression**")
        st.write("A linear model that outputs probabilities using the Sigmoid function. It learns by minimizing **Log Loss (Binary Cross-Entropy)**:")
        st.latex(r"J(\theta) = -\frac{1}{m} \sum \left[ y \log(\hat{y}) + (1-y) \log(1-\hat{y}) \right]")
        st.write("*Pros:* Highly interpretable and fast.")
        
    with col_alg2:
        st.info("**Decision Tree**")
        st.write("A non-linear model that splits data by asking true/false questions. It learns by minimizing **Gini Impurity** at each node:")
        st.latex(r"Gini = 1 - \sum_{i=1}^{C} p_i^2")
        st.write("*Pros:* Captures complex patterns. *Cons:* Prone to overfitting.")
        
    with col_alg3:
        st.info("**Random Forest**")
        st.write("An **Ensemble Method** that builds hundreds of Decision Trees using *Bootstrap Aggregation (Bagging)* and averages their predictions.")
        st.latex(r"\hat{y} = \frac{1}{N} \sum_{i=1}^{N} f_i(x)")
        st.write("*Pros:* Extremely robust, prevents overfitting, best overall accuracy.")

    st.markdown("---")
    st.header("2. Performance Evaluation & The Final Verdict")
    
    col_eval1, col_eval2 = st.columns([1, 1])
    
    with col_eval1:
        st.subheader("Metrics Comparison")
        metrics_df = pd.DataFrame(metrics_dict).T.reset_index()
        metrics_df.rename(columns={'index': 'Model'}, inplace=True)
        st.dataframe(metrics_df.style.highlight_max(subset=['Recall', 'F1_Score', 'Accuracy'], color='lightgreen', axis=0), use_container_width=True)
        
        st.write("**Why Recall is King:** In fraud detection, missing a fraudulent swipe (False Negative) costs the bank money. Blocking a legitimate swipe (False Positive) is just an annoyance. Therefore, we optimize for **Recall**.")
        
    with col_eval2:
        st.subheader("ROC Curve (Receiver Operating Characteristic)")
        st.write("This curve plots the True Positive Rate vs. False Positive Rate. The closer the area under the curve (AUC) is to 1.0, the better the model is at distinguishing between the classes.")
        
        fig_roc = go.Figure()
        fig_roc.add_shape(type='line', line=dict(dash='dash', color='white'), x0=0, x1=1, y0=0, y1=1)
        
        for name in models.keys():
            fpr = roc_data[name]['fpr']
            tpr = roc_data[name]['tpr']
            auc_val = roc_data[name]['auc']
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC = {auc_val:.3f})"))
            
        fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=350, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig_roc, use_container_width=True)

    st.success("🏆 **The Verdict:** **Random Forest** is the superior model for this architecture. While Logistic Regression had slightly higher baseline recall due to SMOTE, Random Forest achieved a vastly superior **F1-Score** and **Precision**, meaning it catches fraud without accidentally blocking thousands of innocent customers.")

# ==========================================
# TAB 3: BATCH INFERENCE & ERROR ANALYSIS
# ==========================================
with tab3:
    st.header("Batch Inference & Error Analysis")
    st.write("""
    Instead of a live simulator, let's look at the hard data. I have loaded a test batch of **10,000 transactions** (including all 492 known frauds). 
    We will run the entire batch through the AI models and analyze exactly where the models succeed and where they fail.
    """)
    
    col_ctrl, col_metric = st.columns([1, 2])
    
    with col_ctrl:
        selected_model_name = st.selectbox("1. Select AI Engine to Process Batch:", list(models.keys()))
        
        st.write("2. Filter the Results Table:")
        filter_choice = st.radio("Show me:", [
            "All 10,000 Transactions", 
            "Caught Fraud (True Positives) ✅", 
            "Missed Fraud (False Negatives) 🚨", 
            "False Alarms (False Positives) ⚠️"
        ])
        
    # --- PROCESS BATCH ---
    model = models[selected_model_name]
    X_batch = batch_df.drop(['Class', 'Time'], axis=1, errors='ignore')[feature_names]
    X_batch_scaled = scaler.transform(X_batch)
    
    probs = model.predict_proba(X_batch_scaled)[:, 1]
    preds = model.predict(X_batch_scaled)
    
    results_df = batch_df[['Time', 'Amount', 'Class']].copy()
    results_df['Fraud Probability'] = np.round(probs * 100, 2).astype(str) + "%"
    results_df['AI Prediction'] = preds
    
    # Classify Outcomes
    conditions = [
        (results_df['Class'] == 1) & (results_df['AI Prediction'] == 1), # TP
        (results_df['Class'] == 0) & (results_df['AI Prediction'] == 1), # FP
        (results_df['Class'] == 1) & (results_df['AI Prediction'] == 0), # FN
        (results_df['Class'] == 0) & (results_df['AI Prediction'] == 0)  # TN
    ]
    choices = [
        'Caught Fraud (True Positives) ✅', 
        'False Alarms (False Positives) ⚠️', 
        'Missed Fraud (False Negatives) 🚨', 
        'Correctly Cleared (True Negatives) ✔️'
    ]
    results_df['Outcome'] = np.select(conditions, choices, default='Unknown')
    
    # Calculate counts
    tp_count = (results_df['Outcome'] == choices[0]).sum()
    fp_count = (results_df['Outcome'] == choices[1]).sum()
    fn_count = (results_df['Outcome'] == choices[2]).sum()
    tn_count = (results_df['Outcome'] == choices[3]).sum()
    
    with col_metric:
        st.subheader(f"Batch Processing Results ({selected_model_name})")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Caught Fraud (TP)", tp_count)
        m2.metric("Missed Fraud (FN)", fn_count, delta="Risk", delta_color="inverse")
        m3.metric("False Alarms (FP)", fp_count, delta="Friction", delta_color="inverse")
        m4.metric("Cleared (TN)", tn_count)
        
    st.markdown("---")
    st.subheader("Data Grid: Transaction Evaluation")
    
    # Filter DataFrame
    if filter_choice != "All 10,000 Transactions":
        display_df = results_df[results_df['Outcome'] == filter_choice]
    else:
        display_df = results_df
        
    # Styling to make it look professional
    def highlight_errors(val):
        if '🚨' in val:
            return 'background-color: #ff4b4b; color: white'
        elif '⚠️' in val:
            return 'background-color: #ffa421; color: black'
        elif '✅' in val:
            return 'background-color: #00CC96; color: white'
        return ''

    styled_df = display_df.style.map(highlight_errors, subset=['Outcome'])
    st.dataframe(styled_df, use_container_width=True, height=400)
