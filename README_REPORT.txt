HOME CREDIT — FP vs TN RAW DATA ANALYSIS
========================================================================
Run timestamp: 20261115_153839
NPZ: /content/drive/MyDrive/Competition/output/CATBOOST_LGBM_XGBOOST_LOCAL_500_NEIGHBOR_INTERACTION/COMPETITION_SUBMISSION/CATBOOST_CREDIT_TERM_PAIRED_EXPERIMENT_20261115_064641/SHAP_OLD_NEW_GLOBAL_20261115_091735/SHAP_PER_CUSTOMER_PAIRED.npz
Raw training CSV: /content/drive/MyDrive/Competition/input/train.csv
Rows in NPZ: 171,202
Features in NEW_SHAP: 33
Decision threshold: 0.5000
FP count: 345 (previous expected: 329)
TN count: 157,036 (previous expected: 41,920)
Folds: [1, 2, 3, 4, 5]

DEFINITIONS
FP = TARGET 0 and predicted default at the selected threshold.
TN = TARGET 0 and predicted non-default at the selected threshold.
All main tests use every FP and every TN; the 5:1 TN sample is for plots only.
Numeric tests: Mann–Whitney within each Fold, combined by Fisher, then Holm correction across features.
Categorical tests: chi-square within each Fold, combined by Fisher, then Holm correction across features.
Effect direction for numeric variables: positive Cliff's delta means FP values tend to be higher than TN.
CREDIT_TERM is AMT_CREDIT / AMT_ANNUITY if not already present.
Interaction tables preserve OOF_FOLD and report FP share within each joint-value cell.

CAUTION
These are associations among TARGET=0 customers partitioned by the model's predictions.
They do not establish that a feature caused the false-positive prediction.
Interpret the fold-adjusted p-values together with effect sizes, fold consistency, and distributions.
Small interaction cells should not be interpreted; see CELL_MEETS_MIN_N.