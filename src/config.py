"""Shared configuration for the Telco Churn project."""

from pathlib import Path

# Reproducibility
SEED = 42
TEST_SIZE = 0.2

# Classification threshold (see notebooks/ML_experiments_decisions.md — favours recall)
THRESHOLD = 0.3

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"

# IBM Telco Customer Churn dataset
TELCO_CHURN_DATA_PATH = RAW_DATA_DIR / "Telco_customer_churn.xlsx"
TELCO_CHURN_SHEET_NAME = "Telco_Churn"

# Target — predict the numeric 0/1 Churn Value directly (no mapping needed).
TARGET = "Churn Value"
ID_COLUMN = "CustomerID"

# Production feature set — matches the validated experiments
# (notebooks/ML_experiments.ipynb). Total Charges is excluded for
# multicollinearity (r=0.83 / VIF>10 with Tenure Months).
NUMERIC_FEATURES = (
    "Zip Code",
    "Latitude",
    "Longitude",
    "Tenure Months",
    "Monthly Charges",
    "CLTV",
)
CATEGORICAL_FEATURES = (
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Allowed categories per categorical column (data contract — single source of
# truth shared by the pandera schema and the Pydantic API model).
CATEGORY_LEVELS = {
    "Gender": ("Female", "Male"),
    "Senior Citizen": ("No", "Yes"),
    "Partner": ("No", "Yes"),
    "Dependents": ("No", "Yes"),
    "Phone Service": ("No", "Yes"),
    "Multiple Lines": ("No", "No phone service", "Yes"),
    "Internet Service": ("DSL", "Fiber optic", "No"),
    "Online Security": ("No", "No internet service", "Yes"),
    "Online Backup": ("No", "No internet service", "Yes"),
    "Device Protection": ("No", "No internet service", "Yes"),
    "Tech Support": ("No", "No internet service", "Yes"),
    "Streaming TV": ("No", "No internet service", "Yes"),
    "Streaming Movies": ("No", "No internet service", "Yes"),
    "Contract": ("Month-to-month", "One year", "Two year"),
    "Paperless Billing": ("No", "Yes"),
    "Payment Method": (
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ),
}

# Plausible numeric ranges (California Telco data, with headroom).
NUMERIC_RANGES = {
    "Zip Code": (90000, 96200),
    "Latitude": (32.0, 42.5),
    "Longitude": (-125.0, -114.0),
    "Tenure Months": (0, 100),
    "Monthly Charges": (0.0, 200.0),
    "CLTV": (0, 10000),
}

# MLflow tracking — sqlite backend (the file store is deprecated in MLflow 3.x).
# Artifacts default to ./mlartifacts; both are gitignored.
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "telco-churn"

# Model artifacts
DEFAULT_PIPELINE_NAME = "model.joblib"
DEFAULT_METADATA_NAME = "meta.json"

# Legacy artifacts — kept for the Etapa 2 MLP notebook (02_neural_network.ipynb).
TARGET_MAPPING = {"No": 0, "Yes": 1}
TOTAL_CHARGES_COLUMN = "Total Charges"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
NEURAL_NETWORK_WEIGHTS_PATH = MODELS_DIR / "churn_mlp.pth"
