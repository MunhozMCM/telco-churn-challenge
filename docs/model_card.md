\# Model Card: Churn Prediction MLP



\## Model Details

\* \*\*Architecture:\*\* Multi-Layer Perceptron (PyTorch) with 2 hidden layers (64 -> 32 neurons) and ReLU activations.

\* \*\*Task:\*\* Binary Classification (Churn vs. Retention).

\* \*\*Optimization:\*\* Adam Optimizer, BCEWithLogitsLoss, trained with Early Stopping (patience=10).



\## Intended Use

\* \*\*Primary Use Case:\*\* Identify telecom customers at high risk of canceling their subscriptions so the retention team can offer targeted discounts.

\* \*\*Out-of-Scope:\*\* This model is not designed to predict churn for enterprise B2B clients, only individual consumers.



\## Metrics \& Performance

\* \*\*Primary Optimization Metric:\*\* F1-Score (Balances False Positives and False Negatives).

\* \*\*Business Metric:\*\* Custo de Churn Evitado. 

\* \*Note: Final metrics are logged dynamically in MLflow (`mlruns/`).\*



\## Limitations \& Biases

\* \*\*Class Imbalance:\*\* The original dataset is heavily skewed towards retained customers (\~73%). Despite stratification, the model may inherently lean towards predicting retention.

\* \*\*Historical Bias:\*\* The model assumes that future churn behavior will exactly mirror historical churn behavior. Market shifts (like a new competitor launching) will degrade accuracy.



\## Failure Scenarios

\* \*\*Extreme Outliers:\*\* Customers with abnormal `TotalCharges` due to billing errors will cause unpredictable network outputs.

\* \*\*Latency Spikes:\*\* Real-time inference relies on FastAPI. High concurrent request volume without load balancing could cause latency timeouts. 



\## Deployment \& Monitoring

\* \*\*Architecture Strategy:\*\* Real-time inference via REST API (FastAPI) to allow immediate integration with CRM systems. 

\* \*\*Monitoring:\*\* Track Data Drift (changes in input distributions like `MonthlyCharges`) and Concept Drift (changes in the actual churn rate over time).

