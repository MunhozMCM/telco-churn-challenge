# ML Experiments — Decision Log

## Decision 1: Lowering the classification threshold to 0.3

### Context

Logistic Regression outputs a churn **probability** per customer, not a hard label. A threshold converts that probability into a binary prediction:

- probability ≥ threshold → predict **Churn (1)**
- probability < threshold → predict **No Churn (0)**

sklearn's default threshold is **0.5** ("more likely than not").

### Problem with the default threshold

The dataset is imbalanced: ~73% of customers did **not** churn. The model learns this prior and becomes conservative — it only fires a churn prediction when it is fairly confident, leaving many real churners with predicted probabilities in the 0.30–0.49 range and classifying them as "No Churn".

Results at threshold = 0.5:

| Metric | Value |
|---|---|
| Recall (Churn) | 0.559 |
| Precision (Churn) | 0.655 |
| F1 (Churn) | 0.603 |
| AUC-ROC | 0.856 |

The high AUC-ROC (0.856) confirms the model **ranks** churners well — it assigns them higher probabilities than non-churners. The weak recall is purely a threshold problem, not a model quality problem.

### Business justification

For a telecom churn use case, the two types of errors have asymmetric costs:

| Error type | What happens | Typical cost |
|---|---|---|
| False Negative (miss a churner) | Customer leaves, revenue lost permanently | High |
| False Positive (flag a non-churner) | Retention offer sent to someone who would have stayed | Low–Medium |

Missing a churner is more costly than wasting a retention offer. Therefore, **higher recall is preferable**, even at the cost of lower precision.

### Chosen threshold: 0.3

Based on the estimated costs (`COST_FN = 500`, `COST_FP = 50`), the cost ratio is 10:1. Theoretically, the optimal threshold that minimizes the expected financial cost would be `~0.09` (`50 / (50 + 500)`).

However, such a low threshold would flag the vast majority of the customer base as likely to churn, generating a volume of false positives that would **make the real operation of the retention team unfeasible** (due to budget and capacity constraints).

Therefore, a threshold of **0.3** was selected to reconcile theoretical mathematics with business practicality: it abandons the default metric (0.5), aggressively prioritizes Recall toward the theoretical optimum, but maintains Precision at an operationally viable level.

---

## Decision 2: Dropping Total Charges due to multicollinearity

### Evidence

Two diagnostics were computed on the numerical features before model training:

**Pearson correlation (r):**

| Pair | r |
|---|---|
| Tenure Months  Total Charges | **0.83** |

**Variance Inflation Factor (VIF):**

| Feature | VIF |
|---|---|
| Total Charges | > 10 (problematic) |
| Tenure Months | elevated |
| Monthly Charges | moderate |
| CLTV | low |

VIF > 10 is the standard statistical threshold indicating that a feature's variance is so inflated by correlation with other features that its coefficient estimates become unreliable.

### Why it matters

Total Charges is largely a function of `Tenure Months × Monthly Charges`. Including it alongside Tenure Months creates redundant information in the feature space. For logistic regression this causes:

- Inflated standard errors on both correlated features
- Generalized instability in the calculated coefficients

### Decision

**Drop Total Charges.** Tenure Months is retained because it is a more direct and interpretable measure of customer loyalty. The information lost is minimal since Monthly Charges (also retained) partially reconstructs the billing signal.

---

## Observation 1: Logistic Regression SHAP value interpretations

After dropping Total Charges, the collinearity distortion was resolved and the SHAP feature importance became clean and interpretable. The following is the business interpretation of each significant feature.

### SHAP values (mean |SHAP|) — Logistic Regression at threshold 0.3

| Feature | Mean \|SHAP\| | Interpretation |
|---|---|---|
| Tenure Months | 0.778 | Strongest predictor. Long-tenured customers rarely churn — loyalty builds over time. Short tenure is the clearest churn signal. |
| Dependents_Yes | 0.577 | Customers with dependents (children, family) churn significantly less. Family setups value service stability and switching costs are higher. |
| Internet Service_Fiber optic | 0.531 | Fiber optic customers churn more. This is likely the most competitive segment where rival providers offer comparable speeds, making switching easier. |
| Contract_Two year | 0.448 | Two-year contracts strongly anchor customers. Contractual commitment is a direct retention mechanism. |
| Contract_One year | 0.228 | One-year contracts also reduce churn relative to month-to-month, but less strongly than two-year. |
| Monthly Charges | 0.218 | Higher monthly bills increase churn risk. Price sensitivity is a secondary driver after loyalty and contract type. |
| Streaming TV_Yes | 0.173 | Moderate positive contribution to churn. Customers using streaming may be more tech-savvy and willing to switch providers. |
| Multiple Lines_Yes | 0.165 | Slight churn signal. May correlate with higher monthly charges. |
| Paperless Billing_Yes | 0.148 | Small positive churn signal. Digitally engaged customers may comparison-shop more actively. |
| Partner_Yes | 0.137 | Having a partner reduces churn slightly. Similar dynamic to dependents — shared accounts have higher switching friction. |
| Online Security_Yes / Tech Support_Yes | ~0.125–0.137 | Customers subscribed to value-added services churn less. They are more embedded in the provider's ecosystem. |

### Features with negligible SHAP contribution (candidates for removal in future iterations)

| Feature | Mean \|SHAP\| | Note |
|---|---|---|
| CLTV | 0.014 | Predicted lifetime value adds almost no signal beyond what tenure and charges already capture. |
| Gender_Male | 0.009 | Gender is not a meaningful churn predictor in this dataset. |
| Online Backup_Yes | 0.007 | Effectively no contribution. |

### Key business takeaways

1. **Retention levers with the highest ROI:** convert month-to-month customers to annual or two-year contracts, and promote value-added services (security, tech support) to increase ecosystem lock-in.
2. **High-risk profile:** new customers (low tenure), on month-to-month contracts, using Fiber optic, without dependents or a partner.
3. **Latitude and Longitude** (0.061 / 0.055) show marginal signal — geographic effects exist but are weak for a linear model. A tree-based model may extract more from spatial data.
