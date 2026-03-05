# Credit Card Fraud Detection Pipeline

A machine learning pipeline for detecting fraudulent credit card transactions, addressing extreme class imbalance and evaluating model performance in terms of financial ROI.

## Overview
This repository contains a comprehensive experiment utilizing the Kaggle European Credit Card Fraud dataset. The project demonstrates the end-to-end ML workflow: data preprocessing, dealing with imbalanced datasets using SMOTE, training multiple classifier architectures, and evaluating them on business-critical metrics.

## Features
- **Imbalance Handling:** Implemented SMOTE (Synthetic Minority Over-sampling Technique) to balance the 0.17% fraud rate during model training.
- **Model Benchmarking:** Head-to-head comparison of Logistic Regression, Decision Trees, Random Forest, and XGBoost.
- **Interpretability:** Feature importance extraction for ensemble models to understand PCA component weighting.
- **ROI Simulation:** Real-world financial impact calculations based on False Negatives (actual money lost) vs. True Positives (money saved).
- **Interactive Dashboard:** A Streamlit application for visualizing the EDA, ROC-AUC curves, and interactive batch inference.

## Technical Stack
- `scikit-learn`: Data splitting, Logistic Regression, Decision Trees, Random Forest, and metric evaluation.
- `xgboost`: Gradient boosting for state-of-the-art tabular classification.
- `imbalanced-learn`: SMOTE implementation.
- `pandas` / `numpy`: Data manipulation.
- `streamlit` / `plotly`: Interactive UI and data visualization.

## Running Locally

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the Kaggle dataset (`creditcard.csv`) inside the `data/` directory.
3. Run the training script to generate the models:
   ```bash
   python ml/train_multiple.py
   ```
4. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

*Author: Vivek*