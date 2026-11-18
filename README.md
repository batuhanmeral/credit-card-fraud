# Credit Card Fraud Detection

An end-to-end machine learning project on the highly imbalanced ULB credit card fraud
dataset (0.17% fraud), built as a learning notebook. Every section of
[`credit_card_fraud.ipynb`](credit_card_fraud.ipynb) starts with an explanation (in Turkish)
of what the step does and why.

## Methods

- **EDA:** structure, missing values, class balance, why accuracy is misleading,
  `Amount` and hour-of-day distributions, features most correlated with `Class`,
  duplicate removal (1,081 rows).
- **Split:** stratified 60% train / 20% validation / 20% test. Model and threshold are
  chosen on validation; the test set is used once at the end.
- **No data leakage:** `RobustScaler` (for `Time` and `Amount`), SMOTE and undersampling
  are fitted on the training set only (resamplers inside an `imblearn` pipeline).
- **Models:** DummyClassifier, Logistic Regression (plain, `class_weight='balanced'`,
  RandomUnderSampler, SMOTE), Random Forest (with and without `class_weight`), XGBoost
  (`scale_pos_weight` from the training set).
- **Evaluation:** confusion matrix, precision, recall, F1, ROC-AUC and PR-AUC, and why
  ROC-AUC looks optimistic on imbalanced data.
- **Threshold selection** on validation: max F1, max precision with recall ≥ 0.85, and
  minimum cost (FN = mean fraud amount in train, FP = 5). The cost-based threshold was
  chosen.
- **Bonus:** SMOTE + `StratifiedKFold` cross-validation inside an `imblearn` pipeline
  (compared with the leaky "SMOTE before CV" approach) and unsupervised anomaly detection
  with IsolationForest.

## Results

Validation set, default threshold 0.5, sorted by PR-AUC:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.930 | 0.851 | 0.889 | 0.978 | **0.890** |
| Random Forest | 0.939 | 0.819 | 0.875 | 0.955 | 0.877 |
| Random Forest + class_weight | 0.908 | 0.840 | 0.873 | 0.960 | 0.876 |
| Logistic Regression | 0.886 | 0.660 | 0.756 | 0.976 | 0.803 |
| LR + SMOTE | 0.050 | 0.904 | 0.095 | 0.977 | 0.788 |
| LR + class_weight | 0.054 | 0.904 | 0.102 | 0.979 | 0.786 |
| LR + RandomUnderSampler | 0.028 | 0.915 | 0.054 | 0.974 | 0.490 |
| Dummy (always "normal") | 0.000 | 0.000 | 0.000 | 0.500 | 0.002 |

**Final result on the test set** (XGBoost, cost-based threshold 0.028 chosen on validation):

| Metric | Value |
|---|---|
| Precision | 0.729 |
| Recall | 0.821 (78 of 95 frauds caught) |
| F1 | 0.772 |
| PR-AUC | 0.832 |
| ROC-AUC | 0.972 |
| Confusion matrix | TP 78 · FP 29 · FN 17 · TN 56,622 |
| Fraud amount caught | 82.3% (12,155.78 of 14,766.31) |

## Dataset

The data is the Kaggle dataset
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(Machine Learning Group, ULB). It is **not** included in this repository. Place the file at:

```
data/creditcard.csv
```

To download it with [kagglehub](https://github.com/Kaggle/kagglehub) (installed by
`requirements.txt`; you may need Kaggle credentials):

```bash
python -c "import kagglehub; kagglehub.dataset_download('mlg-ulb/creditcardfraud', output_dir='data')"
```

## Setup and run

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook credit_card_fraud.ipynb
```

To run the whole notebook from the command line (about 3–4 minutes):

```bash
jupyter nbconvert --to notebook --execute --inplace credit_card_fraud.ipynb
```

## License

[MIT](LICENSE)
