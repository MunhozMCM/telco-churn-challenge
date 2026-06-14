"""Shared configuration for the Telco Churn project."""

from pathlib import Path


# Reproducibility
SEED = 42
TEST_SIZE = 0.2

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

# IBM Telco Customer Churn dataset
TELCO_CHURN_DATA_PATH = RAW_DATA_DIR / "Telco_customer_churn.xlsx"
TELCO_CHURN_SHEET_NAME = "Telco_Churn"
TARGET = "Churn Label"
TARGET_MAPPING = {"No": 0, "Yes": 1}
TOTAL_CHARGES_COLUMN = "Total Charges"

COLUMNS_TO_DROP = (
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason",
    "TotalCharges",
)

# Model artifacts
DEFAULT_PIPELINE_NAME = "model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
NEURAL_NETWORK_WEIGHTS_PATH = MODELS_DIR / "churn_mlp.pth"
