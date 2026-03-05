import os
import pandas as pd
from sklearn.datasets import fetch_openml

def download_openml_data():
    print("Downloading Credit Card Fraud dataset from OpenML...")
    print("This might take a minute as it's a large dataset (284k rows)...")
    
    # OpenML ID 1597 is the credit card fraud dataset
    fraud_data = fetch_openml(data_id=1597, as_frame=True, parser='auto')
    
    df = fraud_data.frame
    
    # Rename 'Class' column to match our code expectations if needed
    # The OpenML dataset usually has 'Class' as string '0' or '1'
    if 'Class' in df.columns:
        df['Class'] = pd.to_numeric(df['Class'])
    
    # Ensure directory exists
    os.makedirs('FraudDetectionGateway/data', exist_ok=True)
    
    # Save to CSV
    csv_path = 'FraudDetectionGateway/data/creditcard.csv'
    df.to_csv(csv_path, index=False)
    print("Successfully downloaded and saved to " + csv_path + "!")
    print("Dataset shape:", df.shape)

if __name__ == "__main__":
    download_openml_data()
