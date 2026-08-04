import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import os
import json

def train_models():
    print("Loading dataset for multi-model training...")
    df = pd.read_csv('data/creditcard.csv')
    
    X = df.drop(['Class', 'Time'], axis=1, errors='ignore') 
    y = df['Class'].astype(int)

    # Split BEFORE scaling so the scaler's mean/std come from training rows
    # only (fitting on the full data leaks test statistics into training).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1, eval_metric='logloss')
    }
    
    os.makedirs('models', exist_ok=True)
    metrics = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        
        metrics[name] = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1_Score": round(f1_score(y_test, y_pred), 4)
        }
        
        model_filename = f"models/{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, model_filename)
        print(f"Finished {name}. Recall: {metrics[name]['Recall']}")
        
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(list(X_train.columns), 'models/features.pkl')
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("All models trained and saved!")

if __name__ == "__main__":
    train_models()
