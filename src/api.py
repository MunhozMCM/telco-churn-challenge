import time
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
import joblib
import torch
import torch.nn as nn
import pandas as pd

# --- 1. SETUP LOGGING ---
import os
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ChurnAPI")

# Setup JSON structured logging for inference requests
request_logger = logging.getLogger("request_logger")
request_logger.setLevel(logging.INFO)
# Save logs inside models directory or a logs dir
_log_path = os.getenv("LOG_ENV_VAR", "models/inference_logs.json")

if not request_logger.handlers:
    fh = logging.FileHandler(_log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    request_logger.addHandler(fh)

# --- 2. REBUILD THE PYTORCH ARCHITECTURE ---
# The API needs to know the shape of the brain before it can load the weights
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

# --- 3. INITIALIZE APP & LOAD ARTIFACTS ---
app = FastAPI(title="Telco Churn Prediction API", version="1.0.0")

# Load artifacts on startup
try:
    # Changed from "../models/..." to "models/..."
    scaler = joblib.load("models/scaler.pkl")
    model = ChurnMLP(input_size=30) 
    model.load_state_dict(torch.load("models/churn_mlp.pth", weights_only=True))
    model.eval() 
    logger.info("Model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load artifacts: {e}")

# --- 4. DTOs (DATA TRANSFER OBJECTS) ---
# This defines exactly what the JSON payload must look like. 
# We'll use a simplified version of our dataset for the endpoint.
class CustomerData(BaseModel):
    tenure: int = Field(..., gt=-1, description="Number of months with the company")
    MonthlyCharges: float = Field(..., gt=0, description="Monthly bill amount")
    TotalCharges: float = Field(..., gt=0, description="Total amount billed")
    Contract_One_year: int = Field(..., ge=0, le=1, description="1 if One Year, 0 otherwise")
    Contract_Two_year: int = Field(..., ge=0, le=1, description="1 if Two Year, 0 otherwise")
    InternetService_Fiber_optic: int = Field(..., ge=0, le=1)
    # Note: A true production API would map ALL 30 features here. 
    # For this challenge MVP, we accept the payload and pad the rest with baseline defaults.

# --- 5. MIDDLEWARE (LATENCY TRACKING) ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Path: {request.url.path} | Method: {request.method} | Latency: {process_time:.4f}s")
    response.headers["X-Process-Time"] = str(process_time)
    return response

# --- 6. ENDPOINTS ---
@app.get("/health")
def health_check():
    """Smoke test endpoint to verify the API is alive."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    """Predicts churn probability for a given customer payload."""
    try:
        # 1. Convert incoming JSON to a dictionary
        input_data = customer.model_dump()
        
        # 2. Map to the 30 features the model expects (Padding missing features with 0)
        # In a real scenario, the payload would contain all 30, or we'd fetch missing data from a DB.
        full_features = {col: 0 for col in range(30)} 
        full_features[0] = input_data['tenure']
        full_features[1] = input_data['MonthlyCharges']
        full_features[2] = input_data['TotalCharges']
        full_features[3] = input_data['Contract_One_year']
        full_features[4] = input_data['Contract_Two_year']
        full_features[5] = input_data['InternetService_Fiber_optic']
        
        # 3. Create DataFrame and Scale
        df_input = pd.DataFrame([full_features.values()])
        scaled_input = scaler.transform(df_input)
        
        # 4. Convert to Tensor and Predict
        tensor_input = torch.tensor(scaled_input, dtype=torch.float32)
        with torch.no_grad():
            output = model(tensor_input)
            probability = torch.sigmoid(output).item()
            
        response_data = {
            "churn_probability": round(probability, 4),
            "risk_level": "High" if probability > 0.5 else "Low"
        }
        
        # Logestruturado
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_payload": input_data,
            "response_payload": response_data,
            "status_code": 200
        }
        request_logger.info(json.dumps(log_record))
        
        return response_data
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}