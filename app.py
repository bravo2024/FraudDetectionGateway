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
    
    for name, model in _models.items():
        y_probs = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        
    return roc_data, y_test

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
roc_data, y_test_true = generate_evaluation_data(df_full, scaler, feature_names, models)
batch_df = get_batch_data(df_full)

# --- UI HEADER ---
st.title("🎓 Machine Learning Concepts: Fraud Detection")
st.markdown("""
*An academic and technical demonstration of Machine Learning concepts, engineered by Vivek (@bravo2024).*  
*Designed to showcase mathematical theory, model evaluation, and deep error investigation.*
""")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Data & Mathematical Theory", "🧠 Algorithms & Industry Context", "🗄️ Error Investigation & Confusion Matrix"])

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
# TAB 2: ALGORITHMS & INDUSTRY CONTEXT
# ==========================================
with tab2:
    st.header("1. The Algorithms & Their Loss Functions")
    st.write("To demonstrate mastery, I trained three different types of baseline algorithms. Here is how they mathematically 'learn' from the data:")
    
    col_alg1, col_alg2, col_alg3 = st.columns(3)
    
    with col_alg1:
        st.info("**Logistic Regression**")
        st.write("A linear model that outputs probabilities using the Sigmoid function. It learns by minimizing **Log Loss (Binary Cross-Entropy)**:")
        st.latex(r"J(\theta) = -\frac{1}{m} \sum \left[ y \log(\hat{y}) + (1-y) \log(1-\hat{y}) \right]")
        
    with col_alg2:
        st.info("**Decision Tree**")
        st.write("A non-linear model that splits data by asking true/false questions. It learns by minimizing **Gini Impurity** at each node:")
        st.latex(r"Gini = 1 - \sum_{i=1}^{C} p_i^2")
        
    with col_alg3:
        st.info("**Random Forest**")
        st.write("An **Ensemble Method** that builds hundreds of Decision Trees using *Bootstrap Aggregation (Bagging)* and averages their predictions.")
        st.latex(r"\hat{y} = \frac{1}{N} \sum_{i=1}^{N} f_i(x)")

    st.markdown("---")
    st.header("2. Performance Evaluation (Baseline Models)")
    
    col_eval1, col_eval2 = st.columns([1, 1])
    
    with col_eval1:
        st.subheader("Metrics Comparison")
        metrics_df = pd.DataFrame(metrics_dict).T.reset_index()
        metrics_df.rename(columns={'index': 'Model'}, inplace=True)
        st.dataframe(metrics_df.style.highlight_max(subset=['Recall', 'F1_Score', 'Accuracy'], color='lightgreen', axis=0), use_container_width=True)
        st.write("**Why Recall is King:** In fraud detection, missing a fraudulent swipe (False Negative) costs the bank money. Blocking a legitimate swipe (False Positive) is just an annoyance. Therefore, we optimize for **Recall**.")
        
    with col_eval2:
        st.subheader("ROC Curve & AUC")
        fig_roc = go.Figure()
        fig_roc.add_shape(type='line', line=dict(dash='dash', color='white'), x0=0, x1=1, y0=0, y1=1)
        for name in models.keys():
            fig_roc.add_trace(go.Scatter(x=roc_data[name]['fpr'], y=roc_data[name]['tpr'], mode='lines', name=f"{name} (AUC = {roc_data[name]['auc']:.3f})"))
        fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=300, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")
    st.header("3. Industry Context: What Do Real Banks Use?")
    st.write("While Random Forest is an excellent educational baseline, enterprise banking systems scale this architecture further using State-of-the-Art (SOTA) models:")
    
    ind1, ind2, ind3 = st.columns(3)
    with ind1:
        st.success("**XGBoost & LightGBM**")
        st.write("*What it is:* Extreme Gradient Boosting. Instead of building independent trees like Random Forest, it builds trees sequentially, where each new tree specifically focuses on correcting the errors of the previous tree.")
        st.write("*Why it's used:* It is currently the undisputed king of tabular data. It handles massive datasets faster and achieves higher precision than Random Forest.")
    with ind2:
        st.warning("**Graph Neural Networks (GNNs)**")
        st.write("*What it is:* Models that treat data as a 'web' or 'graph' rather than isolated rows. Nodes are accounts, edges are transactions.")
        st.write("*Why it's used:* Real-life fraud is committed by organized rings. GNNs catch fraud by noticing that a card sent money to a merchant, who shares an IP address with a known scammer. It tracks the *relationships*.")
    with ind3:
        st.error("**Autoencoders (Deep Learning)**")
        st.write("*What it is:* Unsupervised neural networks designed to compress data into a bottleneck and reconstruct it.")
        st.write("*Why it's used:* For **Anomaly Detection**. Real banks see millions of new types of fraud every day that have no 'labels'. Autoencoders learn what 'normal' looks like, and flag anything it fails to reconstruct properly.")

# ==========================================
# TAB 3: BATCH INFERENCE & ERROR ANALYSIS
# ==========================================
with tab3:
    st.header("Batch Inference & Deep Error Investigation")
    st.write("We have loaded a test batch of **10,000 transactions** (including all 492 known frauds). Let's process the batch and perform a Root Cause Analysis on the AI's mistakes.")
    
    col_ctrl, col_cm = st.columns([1, 1])
    
    with col_ctrl:
        selected_model_name = st.selectbox("1. Select AI Engine to Process Batch:", list(models.keys()))
        
        # --- PROCESS BATCH ---
        model = models[selected_model_name]
        X_batch = batch_df.drop(['Class', 'Time'], axis=1, errors='ignore')[feature_names]
        X_batch_scaled = scaler.transform(X_batch)
        
        probs = model.predict_proba(X_batch_scaled)[:, 1]
        preds = model.predict(X_batch_scaled)
        
        results_df = batch_df[['Amount', 'Class']].copy()
        # Add a mock Transaction ID for display purposes
        results_df.insert(0, 'Transaction_ID', ['TXN-' + str(i).zfill(5) for i in range(len(results_df))])
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
            'Caught Fraud (True Positive) ✅', 
            'False Alarm (False Positive) ⚠️', 
            'Missed Fraud (False Negative) 🚨', 
            'Correctly Cleared (True Negative) ✔️'
        ]
        results_df['Outcome'] = np.select(conditions, choices, default='Unknown')
        
        tp_count = (results_df['Outcome'] == choices[0]).sum()
        fp_count = (results_df['Outcome'] == choices[1]).sum()
        fn_count = (results_df['Outcome'] == choices[2]).sum()
        tn_count = (results_df['Outcome'] == choices[3]).sum()

        st.write("2. Filter the Results Table:")
        filter_choice = st.radio("Show me:", [
            "All 10,000 Transactions", 
            "Caught Fraud (True Positive) ✅", 
            "Missed Fraud (False Negative) 🚨", 
            "False Alarm (False Positive) ⚠️"
        ])
        
    with col_cm:
        st.subheader("The Confusion Matrix")
        st.write("This matrix visually plots exactly where the model succeeded and where it became 'confused'.")
        cm_array = np.array([[tn_count, fp_count], [fn_count, tp_count]])
        
        fig_cm = px.imshow(cm_array, text_auto=True, color_continuous_scale='Blues', aspect='auto',
                           labels=dict(x="What the AI Predicted", y="The True Reality", color="Transactions"),
                           x=['Predicted Legitimate', 'Predicted Fraud'],
                           y=['Actually Legitimate', 'Actually Fraud'])
        fig_cm.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.header("🔍 Investigating the Errors")
    
    col_err1, col_err2 = st.columns(2)
    with col_err1:
        st.error(f"**Why did we miss {fn_count} frauds? (False Negatives)**")
        st.write("False negatives happen when a fraudster perfectly mimics a legitimate customer. Usually, they do 'test charges'—small amounts to see if the card works before draining it. Our model looks at the small amount and normal PCA features, and assumes it's safe.")
    with col_err2:
        st.warning(f"**Why did we block {fp_count} innocent people? (False Positives)**")
        st.write("False positives occur due to 'Anomalous Legitimate Behavior'. If an innocent person suddenly travels to a new country and buys a $3,000 TV, the math mathematically flags it as an extreme anomaly, even though no crime occurred.")

    st.markdown("---")
    
    # Filter DataFrame
    if filter_choice != "All 10,000 Transactions":
        display_df = results_df[results_df['Outcome'] == filter_choice]
    else:
        display_df = results_df
        
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
