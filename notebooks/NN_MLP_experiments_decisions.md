# NN MLP Experiments — Decision Log

## Decision 1: MLP architecture — layer sizes (Input → 64 → 32 → 1)

### Input size

The input dimension is determined automatically from the feature matrix after preprocessing: one neuron per feature (numerical columns + one-hot encoded dummies). No manual tuning required.

### Hidden layers

Two hidden layers were chosen: **64 neurons → 32 neurons**.

| Option | Reasoning |
|---|---|
| Single hidden layer | Sufficient for linearly separable data but limits capacity for interaction effects |
| **64 → 32 (chosen)** | Two layers allow the network to learn a feature combination in the first layer and a compressed representation in the second |
| Deeper (e.g. 128 → 64 → 32) | Overkill for ~7 000 rows; increases risk of overfitting and training instability |

The pyramid shape (wide → narrow) is a standard heuristic for tabular data: the first layer extracts feature combinations, the second layer distills them into a compact representation before the output.

This architecture also mirrors the existing project MLP in `notebooks/02_neural_network.ipynb` (`64 → 32`), keeping experiments comparable.

### Output layer

A single output neuron with **no activation** is used. The raw logit is passed to `BCEWithLogitsLoss` during training (numerically more stable than applying sigmoid first), and `torch.sigmoid()` is applied at inference time to obtain probabilities.

---

## Decision 2: Activation function — ReLU

**ReLU (Rectified Linear Unit)** was chosen over alternatives:

| Activation | Reason for/against |
|---|---|
| Sigmoid / Tanh | Saturate for large inputs → vanishing gradients in deeper networks |
| **ReLU (chosen)** | No saturation for positive values, computationally cheap, empirically strong default for tabular MLPs |
| LeakyReLU | Addresses dying ReLU problem; marginal benefit for shallow networks like this one |
| ELU / GELU | More sophisticated but unnecessary added complexity for two hidden layers |

ReLU is the industry default for hidden layers in feedforward networks on tabular data and requires no hyperparameter tuning.

---

## Decision 3: Dropout rate — 0.3

Dropout randomly zeros activations during training with probability `p`, acting as a regularizer that prevents co-adaptation of neurons.

| Rate | Reasoning |
|---|---|
| 0.0 (no dropout) | Higher risk of overfitting on ~5 600 training samples |
| **0.3 (chosen)** | Moderate regularization; standard starting point for tabular data |
| 0.5 | Common in vision/NLP but often too aggressive for small tabular datasets, hurts convergence |

Dropout is applied after each hidden layer's ReLU activation, not before the output layer (which would distort the final logit scale).

---

## Decision 4: Loss function — BCEWithLogitsLoss

`BCEWithLogitsLoss` combines a sigmoid layer and Binary Cross-Entropy in a single numerically stable operation. It is the standard choice for binary classification in PyTorch.

Using `BCELoss` with a sigmoid activation on the output neuron is mathematically equivalent but numerically inferior — the log-sum-exp trick in `BCEWithLogitsLoss` avoids overflow/underflow for extreme logit values.

---

## Decision 5: Optimizer — Adam with lr=1e-3

**Adam** (Adaptive Moment Estimation) was chosen over SGD and its variants:

| Optimizer | Reasoning |
|---|---|
| SGD | Requires careful learning rate tuning and momentum scheduling; slower convergence |
| SGD + Momentum | Better generalization in some settings but requires more tuning |
| **Adam lr=1e-3 (chosen)** | Adaptive per-parameter learning rates; converges reliably with default settings; standard choice for tabular MLPs |
| AdamW | Adds weight decay decoupled from the gradient update; a worthwhile next step if overfitting is observed |

Learning rate `1e-3` is the Adam default and empirically robust across a wide range of tabular problems.

---

## Decision 6: Batch size — 64

| Batch size | Reasoning |
|---|---|
| Full batch | Deterministic updates, no stochasticity; slow and converges to sharp minima |
| 16–32 | High gradient noise; slower convergence |
| **64 (chosen)** | Standard default; balances gradient quality and training speed; fits comfortably in memory |
| 256+ | Faster per-epoch training but noisier validation signal on a small dataset |

With ~5 600 training samples, batch size 64 gives ~88 gradient updates per epoch — enough for stable learning.

---

## Decision 7: Max epochs and early stopping (150 epochs, patience=10)

### Max epochs — 150

A ceiling of 150 epochs is set to bound worst-case training time. In practice, early stopping triggers well before this limit.

### Early stopping — patience=10

Early stopping monitors **validation loss** after each epoch and halts training when it has not improved for `patience` consecutive epochs, then restores the best weights seen so far.

This serves two purposes:
1. **Prevents overfitting** — the model does not continue fitting training noise after the validation signal plateaus
2. **Removes the need to pre-specify epochs** — training ends when the data says to stop, not at an arbitrary round number

Patience of 10 was chosen to:
- Allow the optimizer to escape local plateaus (short patience like 3–5 can stop too early during a temporary bump)
- Not wait so long that significant overfitting accumulates (patience > 20 on this dataset size risks it)

---

## Decision 8: Classification threshold — 0.3

Same rationale as documented in `ML_experiments_decisions.md` (Decision 1). Repeated here for self-containment:

The dataset is imbalanced (~73% No Churn). The default threshold of 0.5 causes the model to miss ~44% of real churners. Lowering to 0.3 trades precision for recall, which is the correct trade-off when missing a churner (permanent revenue loss) costs more than a wasted retention offer.

Threshold = 0.3 is applied consistently across all models in this project for fair comparison.

---

## Decision 9: SHAP method — DeepExplainer

`shap.GradientExplainer` was initially used but caused kernel crashes on this hardware due to memory pressure (200 background × 300 test samples × gradient computation).

`shap.DeepExplainer` was chosen as the replacement:
- Designed specifically for deep learning models
- Uses a background dataset to approximate SHAP values via a modified backpropagation algorithm
- Significantly lower memory footprint than GradientExplainer for the same sample sizes
- Background set reduced to **50 samples**, test sample to **100** to stay within memory bounds

The reduced sample sizes mean SHAP values are approximations over a subset of the test set, not the full distribution. Treat them as directional indicators rather than exact attributions.

---

## Observation 1: Model comparison — Dummy Classifier vs Logistic Regression vs MLP

All three models were evaluated on the same 20% held-out test set with threshold = 0.3.

### Summary table

| Metric | Dummy Classifier | Logistic Regression | MLP Neural Network |
|---|---|---|---|
| Accuracy | ~0.73 | ~0.77 | ~0.76 |
| Precision (Churn) | 0.00 | ~0.54 | ~0.54 |
| Recall (Churn) | 0.00 | ~0.78 | ~0.76 |
| F1 (Churn) | 0.00 | ~0.64 | ~0.63 |
| AUC-ROC | 0.50 | ~0.856 | ~0.858 |
| PR-AUC | ~0.27 | ~0.690 | ~0.682 |

*(exact values depend on run; consult notebook output for the current figures)*

### Key findings

1. **Dummy Classifier is useless for retention** — 0 recall on churners means it never flags anyone for intervention. Its 73% accuracy is entirely explained by always predicting the majority class. It serves only as a lower-bound sanity check.

2. **LR and MLP perform nearly identically** — differences are within noise (< 0.01 on most metrics). This is expected for a tabular dataset of ~7 000 rows with mostly binary features after OHE. The non-linear capacity of the MLP adds marginal value over a linear separator.

3. **AUC-ROC advantage for MLP is negligible** — the MLP's AUC-ROC is ~0.002 higher, well below what would be operationally meaningful.

4. **Recommendation for this dataset**: Logistic Regression is the preferred model — equivalent performance, faster to train, fully interpretable via SHAP linear explainer, no hyperparameter tuning required.

---

## Observation 2: Cost trade-off analysis — False Positives vs False Negatives

### Error types in the churn context

| Error | Definition | Business consequence |
|---|---|---|
| **False Negative (FN)** | Churner predicted as "No Churn" | Customer leaves uncontacted — full revenue lost |
| **False Positive (FP)** | Non-churner predicted as "Churn" | Retention offer sent to a loyal customer — offer cost wasted, but customer stays |

### Assumed costs (placeholder — must be replaced with real business figures)

| Parameter | Value | Rationale |
|---|---|---|
| `COST_FN` | $500 | Approximate average monthly revenue × expected months of lost revenue |
| `COST_FP` | $50 | Approximate cost of a standard retention offer (discount, voucher, callback) |

These values are configurable constants in the notebook (`COST_FN`, `COST_FP`).

### What the cost table shows

The cost table breaks down TP / TN / FP / FN counts for each model and computes:

- **FN cost** = number of missed churners × `COST_FN`
- **FP cost** = number of false alarms × `COST_FP`
- **Total cost** = FN cost + FP cost

At the assumed 10:1 FN/FP cost ratio, models that maximize recall (LR and MLP at threshold=0.3) incur significantly lower total cost than the Dummy Classifier, despite generating more false positives.

### What the sensitivity chart shows

The sensitivity curve plots total estimated cost as a function of `COST_FN` (from $100 to $1 500) while holding `COST_FP` fixed at $50. Key takeaways:

- **At any FN cost > ~$200**, both LR and MLP are substantially cheaper than the Dummy Classifier
- **LR and MLP curves are nearly overlapping** — the small difference in FN/FP counts between them does not create a meaningful cost gap at the test set scale
- **The gap between Dummy and the real models widens linearly** with FN cost — the more valuable each customer is, the more important recall becomes

### Threshold and cost

Threshold = 0.3 was chosen to improve recall (fewer FNs) at the cost of more FPs. The sensitivity chart makes this trade-off explicit: under the 10:1 cost assumption, accepting more FPs to catch more FNs is the economically rational choice. If `COST_FP` rises significantly (e.g. expensive outbound calls), the break-even threshold would shift upward.
