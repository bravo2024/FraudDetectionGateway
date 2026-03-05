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
st.set_page_config(page_title="Fraud Detection AI Pipeline", page_icon="🎓", layout="wide")

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
st.title("🎓 Comprehensive Experiment on Fraud Detection")
st.markdown("""
*An academic and technical demonstration engineered by Vivek (@bravo2024).*  
*Utilizing the Kaggle European Credit Card Dataset to showcase model evolution, financial ROI, and feature interpretability.*
""")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data & Mathematics", 
    "🧠 Model Evolution (Inc. XGBoost)", 
    "🔍 Feature Importance",
    "💵 Business Impact (ROI & Errors)"
])

# ==========================================
# TAB 1: DATA & MATHEMATICS
# ==========================================
with tab1:
    st.header("1. The Dataset: Origin & Context")
    st.write("**Source:** The official European Credit Card Fraud dataset (Kaggle).")
    st.write("**Volume:** 284,807 total transactions.")
    st.write("**The Problem:** We are looking for **492** fraudulent transactions hidden among **284,315** legitimate ones (0.17% fraud rate).")
    
    st.markdown("---")
    col_math1, col_math2 = st.columns(2)
    with col_math1:
        st.subheader("Data Privacy via PCA")
        st.write("Banks use Principal Component Analysis (PCA) to hide user identities (like location, IP, merchant) while preserving mathematical variance.")
        st.latex(r"Z = XW")
    with col_math2:
        st.subheader("Solving Imbalance via SMOTE")
        st.write("To prevent the AI from defaulting to 'Legitimate', I used SMOTE to algorithmically generate synthetic fraud vectors during training.")
        st.latex(r"x_{new} = x_i + \lambda (x_{zi} - x_i)")

# ==========================================
# TAB 2: MODEL EVOLUTION
# ==========================================
with tab2:
    st.header("Model Evolution & Evaluation")
    st.write("I designed this experiment to track the evolution of algorithms from a simple baseline to State-of-the-Art (SOTA).")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**1. Logistic Regression**\nLinear baseline model. Fast but struggles with complex non-linear patterns.")
    c2.info("**2. Decision Tree**\nRule-based logic. Captures complexity but is highly prone to overfitting the training data.")
    c3.info("**3. Random Forest**\nAn Ensemble method. Builds hundreds of trees to prevent overfitting and increase precision.")
    c4.success("**4. XGBoost (SOTA)**\nExtreme Gradient Boosting. It builds trees sequentially, where each new tree specifically focuses on correcting the errors of the previous tree. The industry standard.")

    st.markdown("---")
    col_eval1, col_eval2 = st.columns([1.2, 1])
    with col_eval1:
        st.subheader("Performance Metrics (Unseen Test Data)")
        metrics_df = pd.DataFrame(metrics_dict).T.reset_index()
        metrics_df.rename(columns={'index': 'Model'}, inplace=True)
        st.dataframe(metrics_df.style.highlight_max(subset=['Recall', 'F1_Score', 'Accuracy'], color='lightgreen', axis=0), use_container_width=True)
    with col_eval2:
        st.subheader("ROC Curve & AUC")
        fig_roc = go.Figure()
        fig_roc.add_shape(type='line', line=dict(dash='dash', color='white'), x0=0, x1=1, y0=0, y1=1)
        for name in models.keys():
            fig_roc.add_trace(go.Scatter(x=roc_data[name]['fpr'], y=roc_data[name]['tpr'], mode='lines', name=f"{name} (AUC = {roc_data[name]['auc']:.3f})"))
        fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=300, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig_roc, use_container_width=True)

# ==========================================
# TAB 3: FEATURE IMPORTANCE
# ==========================================
with tab3:
    st.header("Inside the AI: Feature Importance")
    st.write("Black-box AI is unacceptable in banking. Here, we crack open the **XGBoost** and **Random Forest** brains to see exactly which PCA vectors they use to catch thieves.")
    
    selected_fi_model = st.radio("Select Model to Audit:", ["XGBoost", "Random Forest"])
    
    model = models[selected_fi_model]
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(15) # Top 15
    
    fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                    title=f"Top 15 Most Critical Features ({selected_fi_model})",
                    color='Importance', color_continuous_scale='viridis')
    fig_fi.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
    st.plotly_chart(fig_fi, use_container_width=True)
    
    st.info("💡 **Insight:** Notice how features like `V14` and `V4` consistently rank at the top across models. In the pre-PCA world, these were likely highly indicative behaviors like 'Distance from Home' or 'Transaction Velocity'.")

# ==========================================
# TAB 4: BUSINESS IMPACT (ROI & ERRORS)
# ==========================================
with tab4:
    st.header("Financial Impact & Error Investigation")
    st.write("We process a batch of **10,000 transactions** to calculate the real-world financial ROI of deploying this model.")
    
    selected_model_name = st.selectbox("Select AI Engine to Deploy:", list(models.keys()))
    
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
    choices = ['True Positive ✅', 'False Positive ⚠️', 'False Negative 🚨', 'True Negative ✔️']
    results_df['Outcome'] = np.select(conditions, choices, default='Unknown')
    
    # Financial Math
    caught_fraud_df = results_df[results_df['Outcome'] == 'True Positive ✅']
    missed_fraud_df = results_df[results_df['Outcome'] == 'False Negative 🚨']
    
    money_saved = caught_fraud_df['Amount'].sum()
    money_lost = missed_fraud_df['Amount'].sum()
    
    st.markdown("---")
    st.subheader("💰 Return on Investment (ROI)")
    r1, r2, r3 = st.columns(3)
    r1.metric("Total Fraud Prevented (Money Saved)", f"${money_saved:,.2f}", delta="Profit", delta_color="normal")
    r2.metric("Total Fraud Missed (Money Lost)", f"${money_lost:,.2f}", delta="Loss", delta_color="inverse")
    r3.metric("Fraud Catch Rate", f"{(len(caught_fraud_df)/(len(caught_fraud_df)+len(missed_fraud_df)))*100:.1f}%")

    st.markdown("---")
    st.subheader("The Confusion Matrix")
    cm_array = np.array([
        [(results_df['Outcome'] == 'True Negative ✔️').sum(), (results_df['Outcome'] == 'False Positive ⚠️').sum()],
        [(results_df['Outcome'] == 'False Negative 🚨').sum(), (results_df['Outcome'] == 'True Positive ✅').sum()]
    ])
    fig_cm = px.imshow(cm_array, text_auto=True, color_continuous_scale='Blues', aspect='auto',
                       labels=dict(x="AI Prediction", y="True Reality"),
                       x=['Predicted Legit', 'Predicted Fraud'],
                       y=['Actually Legit', 'Actually Fraud'])
    fig_cm.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_cm, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Deep Error Investigation Grid")
    filter_choice = st.radio("Isolate Outcomes:", ["All Data", "True Positive ✅", "False Negative 🚨", "False Positive ⚠️"])
    
    display_df = results_df if filter_choice == "All Data" else results_df[results_df['Outcome'] == filter_choice]
    
    def highlight(val):
        if '🚨' in val: return 'background-color: #ff4b4b; color: white'
        if '⚠️' in val: return 'background-color: #ffa421; color: black'
        if '✅' in val: return 'background-color: #00CC96; color: white'
        return ''
    st.dataframe(display_df.style.map(highlight, subset=['Outcome']), use_container_width=True, height=300)
