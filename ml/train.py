import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_model():
    print("Starting ML Training Pipeline...")
    
    data_path = 'FraudDetectionGateway/data/creditcard.csv'
    
    if not os.path.exists(data_path):
        print("Error: Dataset not found at " + data_path)
        return

    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Ensure Class is integer
    df['Class'] = df['Class'].astype(int)
    
    # Preprocess
    print("Preprocessing data (scaling & splitting)...")
    
    X = df.drop(['Class'], axis=1) 
    # Drop 'Time' if it exists
    if 'Time' in X.columns:
        X = X.drop(['Time'], axis=1)
        
    y = df['Class']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Applying SMOTE to balance the classes in training data (this may take a moment)...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_resampled, y_train_resampled)
    
    print("Evaluating model on test data...")
    y_pred = model.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save Model and Scaler
    os.makedirs('FraudDetectionGateway/models', exist_ok=True)
    joblib.dump(model, 'FraudDetectionGateway/models/fraud_model.pkl')
    joblib.dump(scaler, 'FraudDetectionGateway/models/scaler.pkl')
    joblib.dump(list(X.columns), 'FraudDetectionGateway/models/features.pkl')
    
    print("Model successfully saved!")

if __name__ == "__main__":
    train_model()
