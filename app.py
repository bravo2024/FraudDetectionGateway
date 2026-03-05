import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fraud Detection Pipeline", layout="wide")

# --- LOAD DATA & MODELS ---
@st.cache_data
def load_sample_data():
    return pd.read_csv('data/creditcard.csv')

@st.cache_data
def compute_correlations(df):
    return df.corr()['Class'].drop('Class').sort_values()

@st.cache_resource
def load_ml_components():
    scaler = joblib.load('models/scaler.pkl')
    features = joblib.load('models/features.pkl')
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    models = {
        "Logistic Regression": joblib.load('models/logistic_regression.pkl'),
        "Decision Tree": joblib.load('models/decision_tree.pkl'),
        "Random Forest": joblib.load('models/random_forest.pkl'),
        "XGBoost": joblib.load('models/xgboost.pkl')
    }
    return scaler, features, metrics, models

@st.cache_data
def generate_evaluation_data(_df_full, _scaler, _features, _models):
    X = _df_full.drop(['Class', 'Time'], axis=1, errors='ignore')[_features]
    y = _df_full['Class'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_scaled = _scaler.transform(X_test)
    
    roc_data = {}
    for name, model in _models.items():
        y_probs = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        
    return roc_data, X_train, y_test

@st.cache_data
def get_batch_data(df):
    frauds = df[df['Class'] == 1]
    legits = df[df['Class'] == 0].sample(n=9508, random_state=42)
    batch = pd.concat([frauds, legits]).sample(frac=1, random_state=42).reset_index(drop=True)
    return batch

df_full = load_sample_data()
correlations = compute_correlations(df_full)
scaler, feature_names, metrics_dict, models = load_ml_components()
roc_data, X_train_raw, y_test_true = generate_evaluation_data(df_full, scaler, feature_names, models)
batch_df = get_batch_data(df_full)

# --- UI HEADER ---
st.title("Credit Card Fraud Detection Pipeline")
st.markdown("""
**Author:** Vivek (GitHub: bravo2024)  
An evaluation of various machine learning models for detecting fraudulent transactions in highly imbalanced datasets, featuring ROI analysis and feature interpretability.
""")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dataset & Preprocessing", 
    "Model Benchmarks", 
    "Feature Interpretability",
    "Financial Impact Analysis",
    "Industry Context"
])

# ==========================================
# TAB 1: DATA & MATHEMATICS
# ==========================================
with tab1:
    st.header("Dataset Context")
    st.write("This project utilizes the European Credit Card Fraud dataset from Kaggle.")
    st.write("- **Total records:** 284,807 transactions")
    st.write("- **Class distribution:** 492 frauds, 284,315 legitimate (0.17% fraud rate)")
    
    st.markdown("---")
    col_math1, col_math2 = st.columns(2)
    with col_math1:
        st.subheader("Feature Anonymization (PCA)")
        st.write("Due to banking privacy regulations, the original features were transformed using Principal Component Analysis (PCA). The resulting dataset consists of numerical principal components (V1-V28), alongside the original 'Time' and 'Amount'.")
        st.latex(r"Z = XW")
    with col_math2:
        st.subheader("Handling Imbalance (SMOTE)")
        st.write("To address the extreme class imbalance (0.17%), the training set was augmented using Synthetic Minority Over-sampling Technique (SMOTE). This generates synthetic examples of the minority class to create a balanced decision boundary.")
        st.latex(r"x_{new} = x_i + \lambda (x_{zi} - x_i)")

# ==========================================
# TAB 2: MODEL EVOLUTION
# ==========================================
with tab2:
    st.header("Algorithm Benchmarks")
    st.write("Four models were trained and evaluated on an isolated 20% test split.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**Logistic Regression**\nStandard linear baseline for binary classification.")
    c2.info("**Decision Tree**\nNon-linear baseline, useful for identifying simple decision rules but prone to overfitting.")
    c3.info("**Random Forest**\nEnsemble approach utilizing bagging to reduce variance and improve generalization.")
    c4.success("**XGBoost**\nGradient boosting framework that builds sequential trees to minimize residual errors. Generally considered SOTA for tabular data.")

    st.markdown("---")
    col_eval1, col_eval2 = st.columns([1.2, 1])
    with col_eval1:
        st.subheader("Test Set Metrics")
        metrics_df = pd.DataFrame(metrics_dict).T.reset_index()
        metrics_df.rename(columns={'index': 'Model'}, inplace=True)
        st.dataframe(metrics_df.style.highlight_max(subset=['Recall', 'F1_Score', 'Accuracy'], color='lightgreen', axis=0), use_container_width=True)
        st.caption("Note: In fraud detection, Recall is prioritized to minimize the financial cost of false negatives.")
    with col_eval2:
        st.subheader("ROC-AUC Curves")
        fig_roc = go.Figure()
        fig_roc.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
        for name in models.keys():
            fig_roc.add_trace(go.Scatter(x=roc_data[name]['fpr'], y=roc_data[name]['tpr'], mode='lines', name=f"{name} (AUC = {roc_data[name]['auc']:.3f})"))
        fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=300, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig_roc, use_container_width=True)

# ==========================================
# TAB 3: FEATURE IMPORTANCE
# ==========================================
with tab3:
    st.header("Model Interpretability")
    st.write("Analyzing which PCA components drive the model's predictions.")
    
    selected_fi_model = st.radio("Select Model:", ["XGBoost", "Random Forest"], horizontal=True)
    
    model = models[selected_fi_model]
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(15)
    
    fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                    title=f"Top 15 Features by Importance ({selected_fi_model})",
                    color='Importance', color_continuous_scale='viridis')
    fig_fi.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
    st.plotly_chart(fig_fi, use_container_width=True)
    
    st.caption("Components like V14 and V4 show strong predictive power across ensemble models. In production, mapping these back to raw feature groups aids in risk policy formulation.")

# ==========================================
# TAB 4: BUSINESS IMPACT
# ==========================================
with tab4:
    st.header("Financial ROI & Error Analysis")
    st.write("Simulating model performance on a batch of 10,000 transactions (including all known fraud cases) to evaluate business impact.")
    
    selected_model_name = st.selectbox("Select Model for Inference:", list(models.keys()))
    
    model = models[selected_model_name]
    X_batch = batch_df.drop(['Class', 'Time'], axis=1, errors='ignore')[feature_names]
    X_batch_scaled = scaler.transform(X_batch)
    
    preds = model.predict(X_batch_scaled)
    
    results_df = batch_df[['Amount', 'Class']].copy()
    results_df.insert(0, 'Transaction_ID', ['TXN-' + str(i).zfill(5) for i in range(len(results_df))])
    results_df['AI Prediction'] = preds
    
    conditions = [
        (results_df['Class'] == 1) & (results_df['AI Prediction'] == 1), # TP
        (results_df['Class'] == 0) & (results_df['AI Prediction'] == 1), # FP
        (results_df['Class'] == 1) & (results_df['AI Prediction'] == 0), # FN
        (results_df['Class'] == 0) & (results_df['AI Prediction'] == 0)  # TN
    ]
    choices = ['True Positive (Caught)', 'False Positive (False Alarm)', 'False Negative (Missed Fraud)', 'True Negative (Cleared)']
    results_df['Outcome'] = np.select(conditions, choices, default='Unknown')
    
    caught_fraud_df = results_df[results_df['Outcome'] == 'True Positive (Caught)']
    missed_fraud_df = results_df[results_df['Outcome'] == 'False Negative (Missed Fraud)']
    
    money_saved = caught_fraud_df['Amount'].sum()
    money_lost = missed_fraud_df['Amount'].sum()
    
    st.markdown("---")
    st.subheader("Simulated Business Metrics")
    r1, r2, r3 = st.columns(3)
    r1.metric("Fraud Prevented (Savings)", f"${money_saved:,.2f}")
    r2.metric("Fraud Missed (Losses)", f"${money_lost:,.2f}")
    
    catch_rate = (len(caught_fraud_df)/(len(caught_fraud_df)+len(missed_fraud_df)))*100 if (len(caught_fraud_df)+len(missed_fraud_df)) > 0 else 0
    r3.metric("Fraud Detection Rate", f"{catch_rate:.1f}%")

    st.markdown("---")
    col_cm, col_grid = st.columns([1, 2])
    
    with col_cm:
        st.subheader("Confusion Matrix")
        cm_array = np.array([
            [(results_df['Outcome'] == 'True Negative (Cleared)').sum(), (results_df['Outcome'] == 'False Positive (False Alarm)').sum()],
            [(results_df['Outcome'] == 'False Negative (Missed Fraud)').sum(), (results_df['Outcome'] == 'True Positive (Caught)').sum()]
        ])
        fig_cm = px.imshow(cm_array, text_auto=True, color_continuous_scale='Blues', aspect='auto',
                           labels=dict(x="Predicted Label", y="True Label"),
                           x=['Legit', 'Fraud'],
                           y=['Legit', 'Fraud'])
        fig_cm.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_grid:
        st.subheader("Transaction Log")
        filter_choice = st.selectbox("Filter by Outcome:", ["All", "True Positive (Caught)", "False Negative (Missed Fraud)", "False Positive (False Alarm)"])
        
        display_df = results_df if filter_choice == "All" else results_df[results_df['Outcome'] == filter_choice]
        
        def highlight(val):
            if 'Missed' in val: return 'background-color: #ff4b4b; color: white'
            if 'False Alarm' in val: return 'background-color: #ffa421; color: black'
            if 'Caught' in val: return 'background-color: #00CC96; color: white'
            return ''
        st.dataframe(display_df.style.map(highlight, subset=['Outcome']), use_container_width=True, height=300)

# ==========================================
# TAB 5: INDUSTRY CONTEXT
# ==========================================
with tab5:
    st.header("Real-World Architecture vs. Kaggle Benchmarks")
    st.write("While XGBoost represents the State-of-the-Art (SOTA) for tabular datasets like this Kaggle challenge, modern financial institutions employ multi-layered, hybrid architectures to combat sophisticated, evolving fraud rings.")
    
    st.markdown("### The Modern Banking AI Stack")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **1. The Rules Engine (Layer 1)**
        Before hitting any ML model, transactions pass through hardcoded business rules (e.g., *'Block if purchase > $5,000 outside home country'*). This filters obvious anomalies instantly with zero compute cost.
        
        **2. XGBoost / LightGBM (Layer 2)**
        The core predictive engine. The model built in this experiment (XGBoost) is exactly what sits at Layer 2 in institutions like Visa or Mastercard. It evaluates historical tabular data in milliseconds.
        """)
        
    with c2:
        st.markdown("""
        **3. Graph Neural Networks (GNNs) (Layer 3)**
        Used by companies like Stripe or PayPal. Instead of looking at isolated rows, GNNs map relationships. If Account A sends money to Merchant B, who shares an IP address with Scammer C, the GNN flags the entire network.
        
        **4. Autoencoders (Unsupervised Deep Learning)**
        Supervised models (like Random Forest) only catch fraud they have seen before. Banks use Autoencoders to detect *Zero-Day Fraud*—brand new techniques. It learns what 'normal' behavior looks like and flags anything mathematically alien.
        """)
        
    st.info("**Why the discrepancy?** Public datasets (like this one) provide isolated, static rows to protect privacy. Real-world systems ingest relational data (IPs, geolocations, device fingerprints) continuously, requiring the hybrid GNN/Boosting architecture described above.")
