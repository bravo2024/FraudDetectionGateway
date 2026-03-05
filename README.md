# 🛡️ Real-Time Fraud Detection Gateway

A live, public-facing dashboard simulating an enterprise Machine Learning pipeline. It evaluates credit card transactions in real-time, detecting anomalies and visualizing the data stream.

## 🌟 Features
- **Live Stream Simulation:** Generates synthetic transactions continuously.
- **Real-Time Inference:** Uses a Machine Learning model (Random Forest) to classify transactions as legitimate or fraudulent instantly.
- **Dynamic Dashboard:** Built entirely in Streamlit with auto-updating metrics, live data tables, and interactive Plotly charts.
- **Ready for Streamlit Cloud:** Easily deployable to the public without requiring a VPS or complex backend architecture.

## 🚀 How to Run Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard:**
   ```bash
   streamlit run app.py
   ```

## 🧠 ML Pipeline (For the Developer)
The `ml/` directory contains the training scripts. The model is trained on the Kaggle Credit Card Fraud dataset, utilizing **SMOTE** to handle extreme class imbalances (99.8% legitimate vs 0.2% fraud).

---
*Built with precision.*