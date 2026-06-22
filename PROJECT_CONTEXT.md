# Project Context

Last verified: 2026-06-22

This file is a working reference for future development in this repository. It records the project facts, local setup, conventions, and constraints that have already been read or established.

## Repository

- Root path: `/home/gabe/workspace/telco-churn-challenge`
- Upstream remote: `origin git@github.com:MunhozMCM/telco-churn-challenge.git`
- Fork remote: `fork git@github.com:Gabrielloufi/telco-churn-challenge.git`
- Current main commit when this file was created: `089395c` (`Merge pull request #1 from Gabrielloufi/feat/data-io-eda-scaffold`)
- Current local status when this file was created: only `notebooks/mlflow.db` was modified locally.
- Do not commit generated/runtime artifacts such as `notebooks/mlflow.db`, `mlruns/`, `models/`, caches, or virtual environments unless explicitly requested.

## User Working Preference

- Keep edits strictly scoped to what is requested.
- Do not add notebook sections, charts, analysis, refactors, or cleanup unless the user asks for them.
- When editing notebooks, preserve existing cells and outputs unless the requested change requires otherwise.
- If a requested notebook change needs a cell added, add only that cell.

## Environment

- Python environment: `notebooks/.venv`
- VS Code interpreter setting: `${workspaceFolder}/notebooks/.venv/bin/python`
- Jupyter kernel display name: `Python (telco-churn-challenge)`
- The repository should be opened in VS Code as `/home/gabe/workspace/telco-churn-challenge`, not `/home`, so `.vscode/settings.json` applies.
- Dependencies are declared in `pyproject.toml`; `requirements.txt` is older and does not list every package used by notebooks.

Primary dependencies currently declared:

- `pandas`
- `scikit-learn`
- `mlflow`
- `torch`
- `fastapi`
- `uvicorn`
- `pytest`
- `ruff`
- `pandera`
- `seaborn`
- `openpyxl`
- `matplotlib`
- `joblib`
- `pydantic`

## Dataset

- Dataset: IBM Telco Customer Churn.
- File path: `data/raw/Telco_customer_churn.xlsx`
- Excel sheet: `Telco_Churn`
- Shape verified locally: `7043` rows by `33` columns.
- Target column: `Churn Label`
- Target values verified locally:
  - `No`: `5174`
  - `Yes`: `1869`
- Distinct `City` values verified locally: `1129`
- Loader function: `src.data.io.load_telco_churn()`
- Preferred DataFrame variable name in project code/notebooks: `telco_churn_df`

Dataset columns:

```text
CustomerID, Count, Country, State, City, Zip Code, Lat Long, Latitude,
Longitude, Gender, Senior Citizen, Partner, Dependents, Tenure Months,
Phone Service, Multiple Lines, Internet Service, Online Security,
Online Backup, Device Protection, Tech Support, Streaming TV,
Streaming Movies, Contract, Paperless Billing, Payment Method,
Monthly Charges, Total Charges, Churn Label, Churn Value, Churn Score,
CLTV, Churn Reason
```

## Source Modules

Current `src/` structure:

```text
src/
├── __init__.py
├── api.py
├── config.py
├── data/
│   ├── __init__.py
│   └── io.py
└── modeling/
    ├── __init__.py
    ├── pipeline.py
    ├── preprocessing.py
    └── train.py
```

`src/config.py` centralizes:

- `SEED = 42`
- `TEST_SIZE = 0.2`
- project, data, raw data, and models paths
- Telco workbook path and sheet name
- target metadata:
  - `TARGET = "Churn Label"`
  - `TARGET_MAPPING = {"No": 0, "Yes": 1}`
  - `TOTAL_CHARGES_COLUMN = "Total Charges"`
- current preprocessing drop-list:
  - `CustomerID`
  - `Count`
  - `Country`
  - `State`
  - `City`
  - `Zip Code`
  - `Lat Long`
  - `Latitude`
  - `Longitude`
  - `Churn Value`
  - `Churn Score`
  - `CLTV`
  - `Churn Reason`
  - `TotalCharges`
- artifact names/paths:
  - `DEFAULT_PIPELINE_NAME = "model.joblib"`
  - `SCALER_PATH = MODELS_DIR / "scaler.pkl"`
  - `NEURAL_NETWORK_WEIGHTS_PATH = MODELS_DIR / "churn_mlp.pth"`

`src/data/io.py` currently provides:

- `load_telco_churn() -> pd.DataFrame`
- `save_pipeline(pipeline, name="model.joblib") -> None`
- `load_pipeline(name="model.joblib") -> Any`

Security note: `joblib.load()` should only load trusted local artifacts.

The `src/modeling/` files are currently placeholders. They exist to match the FIAP textbook structure and are not implemented yet.

## API

`src/api.py` is the current FastAPI entry point.

Current behavior:

- Defines `ChurnMLP`, a PyTorch model with hidden layers `64 -> 32` and ReLU activations.
- Loads:
  - `models/scaler.pkl`
  - `models/churn_mlp.pth`
- Exposes:
  - `GET /health`
  - `POST /predict`
- Uses Pydantic model `CustomerData`.
- The current API payload is simplified and pads missing features with zeroes to fit a 30-feature model input.
- `src/api.py` predates the new config module and still uses direct artifact paths.
- Ruff format check reports `src/api.py` would be reformatted; do not reformat it unless requested because it is unrelated to the current scaffold work.

## Notebooks

Notebook kernel metadata should use:

```json
{
  "display_name": "Python (telco-churn-challenge)",
  "language": "python",
  "name": "telco-churn-challenge"
}
```

Current notebooks:

- `notebooks/01_eda_and_baselines.ipynb`
  - Baseline EDA and classical model workflow.
  - Contains direct `pd.read_excel(...)` reads.
  - Some cells still use legacy `../data/Telco_customer_churn.xlsx` paths; do not alter unless asked.
- `notebooks/02_neural_network.ipynb`
  - PyTorch MLP training workflow.
  - Uses `StandardScaler`, `DataLoader`, MLflow metrics, and saves artifacts to `../models/scaler.pkl` and `../models/churn_mlp.pth`.
- `notebooks/exploratory_Eda.ipynb`
  - Created as a working exploratory notebook.
  - It currently mirrors the baseline notebook plus one explicitly requested cell counting distinct `City` values.

Known notebook/Jupyter notes:

- VS Code Jupyter extension version observed: `2025.9.1`.
- A VS Code Jupyter UI state bug was observed where kernels remained responsive but cells appeared stuck.
- If notebooks freeze, first check for stale kernels and open the repository root directly in VS Code.

## Model Card

`docs/model_card.md` describes:

- Model: PyTorch MLP, binary churn classification.
- Architecture: two hidden layers, `64 -> 32`, ReLU.
- Training: Adam, `BCEWithLogitsLoss`, early stopping with patience `10`.
- Main metric: F1 score.
- Business goal: identify likely churn customers for retention actions.
- Known limitations:
  - class imbalance, around 73% retained customers
  - historical behavior may not represent future churn behavior
  - outlier billing values can destabilize predictions
  - API latency can degrade under load without proper serving infrastructure

## FIAP Reference Material

Local path:

```text
material-fiap/FIAP/engenharia-api
```

This material is ignored by Git via `.gitignore`.

Relevant guidance from the FIAP textbook:

- Refactor notebook logic into `src/` modules.
- Keep `data/raw` immutable.
- Use `src/config.py` for constants and seeds.
- Use `src/data/io.py` for external I/O and model serialization.
- Use `src/modeling/preprocessing.py`, `pipeline.py`, and `train.py` for production ML logic.
- Use tests, schema validation, FastAPI contracts, and tooling as the project matures.
- The practical `codigos/` implementation advertised in the FIAP README is not present in the cloned material.

## Validation Commands

Useful local checks:

```bash
notebooks/.venv/bin/python -m py_compile \
  src/__init__.py \
  src/config.py \
  src/data/__init__.py \
  src/data/io.py \
  src/modeling/__init__.py \
  src/modeling/pipeline.py \
  src/modeling/preprocessing.py \
  src/modeling/train.py
```

```bash
notebooks/.venv/bin/ruff check src
```

Targeted loader check:

```bash
notebooks/.venv/bin/python - <<'PY'
from src.data.io import load_telco_churn

telco_churn_df = load_telco_churn()
assert telco_churn_df.shape == (7043, 33)
assert telco_churn_df["City"].nunique(dropna=False) == 1129
print("loader ok")
PY
```

There are currently no pytest tests in the repository.

## Git Hygiene

Ignored by `.gitignore`:

- `venv/`
- `.venv/`
- `.env`
- `__pycache__/`
- `*.pyc`
- `/data/`
- `/models/`
- `mlruns/`
- `material-fiap/`

Generated local files seen during work:

- `.pytest_cache/`
- `.ruff_cache/`
- `src/telco_churn_challenge.egg-info/`
- `notebooks/mlflow.db`

Do not revert user changes unless explicitly asked. If a file is dirty and unrelated to the requested task, leave it alone.
