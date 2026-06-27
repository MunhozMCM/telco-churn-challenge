import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import mlflow

os.makedirs('models', exist_ok=True)

class ChurnMLP(nn.Module):
    def __init__(self, input_size):
        super(ChurnMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        out = self.relu1(self.layer1(x))
        out = self.relu2(self.layer2(out))
        return self.output_layer(out)

def preprocess_data(df):
    drop_cols = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 
                 'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 'Churn Score', 
                 'CLTV', 'Churn Reason']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Clean Total Charges
    df['Total Charges'] = pd.to_numeric(df['Total Charges'].replace(' ', np.nan))
    df = df.dropna()
    
    y = df['Churn Value'].values
    X = df.drop(columns=['Churn Value'])
    
    # Get dummies
    X = pd.get_dummies(X, drop_first=True)
    X = X.astype(float)
    
    # Reorder columns so that the first ones match the API logic
    # api.py indices: 0: tenure, 1: MonthlyCharges, 2: TotalCharges, 
    # 3: Contract_One year, 4: Contract_Two year, 5: Internet Service_Fiber optic
    important_cols = [
        'Tenure Months', 'Monthly Charges', 'Total Charges', 
        'Contract_One year', 'Contract_Two year', 'Internet Service_Fiber optic'
    ]
    # Ensure they exist
    for col in important_cols:
        if col not in X.columns:
            X[col] = 0.0
            
    other_cols = [c for c in X.columns if c not in important_cols]
    final_cols = important_cols + other_cols
    X = X[final_cols]
    
    # Save column names to a file for reference
    joblib.dump(final_cols, 'models/feature_columns.pkl')
    
    return X.values, y, final_cols

def train_baselines(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="Dummy_Baseline"):
        dummy = DummyClassifier(strategy='stratified', random_state=42)
        dummy.fit(X_train, y_train)
        preds = dummy.predict(X_test)
        f1 = f1_score(y_test, preds)
        mlflow.log_metric("f1_score", f1)
        print(f"Dummy F1: {f1:.4f}")
        
    with mlflow.start_run(run_name="Logistic_Regression"):
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        preds = lr.predict(X_test)
        f1 = f1_score(y_test, preds)
        mlflow.log_metric("f1_score", f1)
        print(f"LR F1: {f1:.4f}")

def train_mlp(X_train, y_train, X_test, y_test, input_size):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    model = ChurnMLP(input_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    with mlflow.start_run(run_name="PyTorch_MLP"):
        mlflow.log_param("epochs", 100)
        mlflow.log_param("lr", 0.001)
        
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
        torch.save(model.state_dict(), 'models/churn_mlp.pth')
        mlflow.log_artifact('models/churn_mlp.pth')
        print("PyTorch MLP trained and saved.")

def main():
    print("Loading data...")
    df = pd.read_excel('data/Telco_customer_churn.xlsx')
    
    X, y, feature_cols = preprocess_data(df)
    print(f"Dataset shape: {X.shape}. Expected 30 features, got {X.shape[1]}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("Training Baselines...")
    mlflow.set_experiment("Telco_Churn")
    train_baselines(X_train_scaled, y_train, X_test_scaled, y_test)
    
    print("Training MLP...")
    train_mlp(X_train_scaled, y_train, X_test_scaled, y_test, input_size=X.shape[1])
    
if __name__ == "__main__":
    main()
