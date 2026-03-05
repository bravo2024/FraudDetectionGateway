# 🛡️ Enterprise Fraud Detection Gateway

**Architected by Vivek (@bravo2024)**

An end-to-end Machine Learning pipeline designed to evaluate highly imbalanced credit card transactions in real-time. This project moves beyond simple predictive modeling to incorporate **Enterprise MLOps** and **Business Logic Tuning**, simulating a true banking environment.

## 🚀 Live Application
View the interactive dashboard here: **[Link to your deployed Streamlit app]**

## 🧠 Core Engineering Architecture

### 1. The Imbalance Problem (SMOTE)
Real-world fraud is extremely rare (0.17% in this dataset). Naive models simply learn to predict "Legitimate" 100% of the time to achieve high accuracy. 
*   **My Solution:** I engineered a preprocessing pipeline utilizing **SMOTE** (Synthetic Minority Over-sampling Technique) to mathematically generate synthetic fraudulent vectors during training. This forces the algorithms to establish a proper decision boundary.

### 2. Multi-Algorithm Benchmarking
I trained and benchmarked three distinct architectures:
*   **Logistic Regression:** Fast, interpretable baseline.
*   **Decision Tree:** Rule-based logic capable of capturing non-linear relationships.
*   **Random Forest:** An ensemble method that provided the most robust resistance to overfitting.
*   *Note: In this domain, I optimized for **Recall** over Accuracy, as the business cost of missing fraud (False Negative) is exponentially higher than a false alarm.*

### 3. Dynamic Business Threshold Tuning
AI does not output binary "Yes/No" answers; it outputs probabilities. The dashboard includes an interactive **Decision Threshold Slider**.
*   This demonstrates my understanding of the **Precision-Recall Tradeoff**. 
*   Users can manipulate the threshold to see exactly how moving the slider impacts "Customer Friction" (blocking innocent people) versus "Business Risk" (letting thieves through).

## 🛠️ Tech Stack
*   **Language:** Python 3
*   **Machine Learning:** Scikit-Learn, Imbalanced-Learn (SMOTE)
*   **Data Processing:** Pandas, NumPy
*   **Visualization:** Streamlit, Plotly
*   **Data Security:** PCA (Principal Component Analysis) to ensure GDPR/banking compliance.

---
*Developed as a portfolio piece to demonstrate applied Machine Learning Engineering.*