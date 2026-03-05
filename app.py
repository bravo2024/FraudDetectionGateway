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

@st.cache_data
def compute_correlations(df):
    # Compute correlation with Class
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

df_full = load_sample_data()
correlations = compute_correlations(df_full)
scaler, feature_names, metrics_dict, models = load_ml_components()

# --- UI HEADER ---
st.title("🛡️ Enterprise Fraud Detection Gateway")
st.markdown("""
*An end-to-end Machine Learning pipeline engineered by Vivek (@bravo2024).*  
*Showcasing Deep Mathematical EDA, Model Benchmarking, and a Live Interactive Inference Engine.*
""")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Data & Mathematical Analysis", "🤖 Model Benchmarking", "🎚️ Live Inference Engine"])

# ==========================================
# TAB 1: DATA & MATHEMATICAL ANALYSIS
# ==========================================
with tab1:
    st.header("1. The Dataset: Origin & Context")
    st.write("""
    **Source:** The official European Credit Card Fraud dataset (ULB Machine Learning Group).  
    **Volume:** 284,807 total transactions processed over exactly 48 hours in September 2013.  
    **The Target:** We are looking for **492** fraudulent transactions hidden among **284,315** legitimate ones.
    """)
    
    st.markdown("---")
    st.header("2. The Mathematical Foundations")
    
    col_math1, col_math2 = st.columns(2)
    
    with col_math1:
        st.subheader("Data Privacy via PCA")
        st.write("""
        Banks cannot legally release names, locations, or merchant details. To preserve the statistical variance while hiding identities, they apply **Principal Component Analysis (PCA)**. 
        """)
        st.latex(r"Z = XW")
        st.write("""
        Where $X$ is the original data, $W$ is the matrix of eigenvectors, and $Z$ represents the new orthogonal features ($V1$ through $V28$). This mathematical transformation is why our features look like numbers instead of names.
        """)
        
    with col_math2:
        st.subheader("Solving Imbalance via SMOTE")
        st.write("""
        With only 0.17% fraud, models will collapse. I utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to algorithmically generate new fraud vectors rather than duplicating old ones.
        """)
        st.latex(r"x_{new} = x_i + \lambda (x_{zi} - x_i)")
        st.write("""
        It selects a fraudulent point $x_i$, finds its $k$-nearest neighbors (like $x_{zi}$), and creates a synthetic vector $x_{new}$ randomly along the line connecting them ($\lambda \in [0,1]$).
        """)
        
    st.markdown("---")
    st.header("3. Deep Analysis on Complete Data")
    st.write("How does the AI actually tell the difference between Fraud and Legitimate? It looks at **Correlations**.")
    
    col_corr1, col_corr2 = st.columns([1, 1.5])
    
    with col_corr1:
        st.subheader("Feature Correlation with Fraud")
        st.write("These are the PCA features that have the strongest mathematical relationship to the `Class` variable.")
        
        # Get top 5 negative and top 5 positive
        top_neg = correlations.head(5)
        top_pos = correlations.tail(5)
        top_corr = pd.concat([top_neg, top_pos]).reset_index()
        top_corr.columns = ['Feature', 'Correlation']
        
        fig_corr = px.bar(top_corr, x='Correlation', y='Feature', orientation='h', 
                          color='Correlation', color_continuous_scale='RdBu')
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with col_corr2:
        st.subheader("Distribution Analysis: V14 vs V4")
        st.write("Let's isolate the most negatively correlated feature (`V14`) and the most positively correlated feature (`V4`) and map them. Notice how the Red (Fraud) points cluster differently than Green (Legit).")
        
        # Sample data for scatter to prevent browser crash
        # Take all fraud, and a random sample of 2000 legit
        fraud_points = df_full[df_full['Class'] == 1]
        legit_points = df_full[df_full['Class'] == 0].sample(n=2000, random_state=42)
        scatter_data = pd.concat([fraud_points, legit_points])
        scatter_data['Label'] = scatter_data['Class'].map({0: 'Legitimate', 1: 'Fraudulent'})
        
        fig_scatter = px.scatter(scatter_data, x='V14', y='V4', color='Label', 
                                 color_discrete_map={'Legitimate': '#00CC96', 'Fraudulent': '#EF553B'},
                                 opacity=0.6)
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

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
            
            X_input = row.drop(['Class', 'Time'], axis=1, errors='ignore')[feature_names]
            X_scaled = scaler.transform(X_input)
            
            model = models[selected_model_name]
            probabilities = model.predict_proba(X_scaled)[0]
            
            fraud_prob = probabilities[1] * 100
            is_blocked = fraud_prob >= threshold
            
            st.markdown("---")
            st.markdown("### 🧠 AI Evaluation & Business Logic")
            
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
