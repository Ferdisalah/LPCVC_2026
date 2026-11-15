# -*- coding: utf-8 -*-
"""
HOME CREDIT — FOUR-MODEL OPTUNA ENSEMBLE (WINDOWS / CUDA)
Fold-local EXT_SOURCE_1/2/3 imputation + nine shared features + cached GMM.

Models:
    1. RealMLP-TD (CUDA GPU)
    2. CatBoost (CUDA GPU)
    3. LightGBM (CPU)
    4. XGBoost (CUDA GPU)

Fixed design:
    - StratifiedKFold(5, shuffle=True, random_state=42).
    - Reuse the saved IMPUTE_EXT_SOURCE_1_2_3 training imputations and the
      matching fold-specific CatBoost imputers for test rows.
    - Keep three imputed EXT_SOURCE values and three original missing flags.
    - Keep GOODS_PRICE_TO_ANNUITY ONLY in the saved imputer input schema;
      it is deliberately excluded from both classifier feature views.
    - Add the nine shared classifier features after EXT_SOURCE imputation.
    - Add signed DIFF_RATIO, NET_INCOME_PER_MEMBER and CAR_FRESHNESS to both views.
    - Build hierarchical CAREER_PROFILE from income, occupation and organization;
      category support is fit on the active training split only.
    - Fit/load one fold-local hybrid LOAN_STRUCTURE_CLUSTER segmenter per fold,
      using the persistent GMM cache scheme from the supplied GMM script.
    - Compute LOCAL_EXT_SOURCE_AGE_INTERACTION once per fold using imputed
      EXT_SOURCE values and a fold-training-only 500-neighbour reference.
    - Keep separate Tree and RealMLP feature views.
    - Optuna tunes all four models with 5 inner folds inside each outer-training fold.
    - RealMLP epoch selection uses a separate split inside inner-training, then refits
      on all outer-training rows at a newly selected epoch.
    - RealMLP/CatBoost/XGBoost use CUDA; LightGBM uses CPU; equal ensemble weights.
    - Atomic checkpoints for each preprocessing stage and model/fold.

IMPORTANT: This is one complete, directly executable Python script. Edit only the
PATHS section below if your Windows paths change. The GMM cache path is:
DATA_DIR / "HOME_CREDIT_GMM_MODEL_CACHE".

Validation protocol:
- Outer StratifiedKFold(5) creates independent validation predictions.
- Optuna uses inner StratifiedKFold(5) only inside each outer-training fold.
- RealMLP epoch selection uses a separate 80/20 split inside each inner-training
  partition; the final outer-fold epoch is selected on an 80/20 split of
  Outer Training, then refit on all Outer Training rows.
- Outer Validation is used only for final model evaluation and OOF prediction.
"""

from __future__ import annotations

# Preserve an explicitly configured CUDA visibility setting; default to GPU 0.
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import gc
import hashlib
import json
import pickle
import subprocess
import sys
import time
import warnings
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier
from packaging.version import Version
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PATHS — EDIT THESE FOR YOUR COMPUTER
# =============================================================================

# Main directory containing your input folder and experiment folders.
DATA_DIR = Path(r"C:\Users\UNITECH\Downloads")

# Raw Home Credit data in the Downloads root (the exact paths requested).
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test (1).csv"
SAMPLE_SUB_FILE = DATA_DIR / "sample_submission.csv"  # optional

# Existing saved EXT_SOURCE imputation experiment directory.
IMPUTATION_EXPERIMENT_DIR = (
    DATA_DIR / "CATBOOST_FULLDATA_TWO_RATIO_FEATURES_EXT_SOURCE_IMPUTATION_GRID"
)
IMPUTATION_MODELS_DIR = IMPUTATION_EXPERIMENT_DIR / "IMPUTATION_MODELS"
IMPUTATION_CLASSIFIER_SCENARIO_DIR = (
    IMPUTATION_EXPERIMENT_DIR / "SCENARIOS" / "IMPUTE_EXT_SOURCE_1_2_3"
)

# Persistent GMM cache path copied from your previous GMM script.
# The script loads a matching cached fold segmenter or fits/saves a new one.
GMM_CACHE_DIR = DATA_DIR / "HOME_CREDIT_GMM_MODEL_CACHE"

# Stable output path: rerunning resumes completed feature/model fold checkpoints.
OUTPUT_DIR = (
    DATA_DIR
    / "output"
    / "HOME_CREDIT_FOUR_MODEL_CPU_IMPUTED_EXT_SOURCE_1_2_3_PLUS_9_SHARED_GMM"
)

# The following folders are generated automatically.
CACHE_DIR = OUTPUT_DIR / "FEATURE_CACHE"
IMPUTATION_TEST_DIR = OUTPUT_DIR / "IMPUTATION_TEST_VALUES"
GMM_DIR = OUTPUT_DIR / "GMM_SEGMENTATION"
MODEL_DIR = OUTPUT_DIR / "MODELS"
PREDICTION_DIR = OUTPUT_DIR / "FOLD_PREDICTIONS"
IMPORTANCE_DIR = OUTPUT_DIR / "FEATURE_IMPORTANCE"
CHECKPOINT_DIR = OUTPUT_DIR / "CHECKPOINTS"
ENSEMBLE_DIR = OUTPUT_DIR / "ENSEMBLE"

for _directory in [
    OUTPUT_DIR,
    CACHE_DIR,
    IMPUTATION_TEST_DIR,
    GMM_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    IMPORTANCE_DIR,
    CHECKPOINT_DIR,
    ENSEMBLE_DIR,
    GMM_CACHE_DIR,
]:
    _directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. FIXED CONFIGURATION
# =============================================================================

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
RANDOM_SEED = 42
N_SPLITS = 5
EXT_SOURCE_COLUMNS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
EXT_SOURCE_MISSING_COLUMNS = [f"{column}_MISSING" for column in EXT_SOURCE_COLUMNS]

MISSING_CATEGORY = "__MISSING__"
LOAN_STRUCTURE_FEATURE = "LOAN_STRUCTURE_CLUSTER"
NEW_INTERACTION_FEATURE = "LOCAL_EXT_SOURCE_AGE_INTERACTION"
CAREER_PROFILE_FEATURE = "CAREER_PROFILE"
CAREER_PROFILE_MIN_COUNT = 40
CAREER_PROFILE_PENSIONER_VALUE = "PENSIONER_RETIRED"

# The nine shared classifier features requested in the CPU/imputation script.
EXT_SOURCE_SUMMARY_FEATURES = [
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_STD",
    "EXT_SOURCE_MIN",
    "EXT_SOURCE_MAX",
    "EXT_SOURCE_PRODUCT",
]
FINANCIAL_AGE_RATIO_FEATURES = [
    "ANNUITY_TO_INCOME",
    "CREDIT_TO_INCOME",
    "EMPLOYED_TO_AGE",
    "DAYS_ID_TO_BIRTH",
]
REQUESTED_SHARED_FEATURES = (
    EXT_SOURCE_SUMMARY_FEATURES + FINANCIAL_AGE_RATIO_FEATURES
)

# CPU thread budget for LightGBM and data-processing tasks. Keep at 2 initially.
CPU_THREADS = 2
CPU_THREADS = max(1, min(CPU_THREADS, os.cpu_count() or CPU_THREADS))

N_NEIGHBORS = 500
NEIGHBOR_QUERY_CHUNK_SIZE = 2048

# Fixed tree-model parameters from the supplied best-parameters configuration.
FIXED_CATBOOST_PARAMS = {
    "iterations": 4200,
    "learning_rate": 0.030207171090683906,
    "depth": 4,
    "l2_leaf_reg": 18.17422221879975,
    "random_strength": 0.6746151640000321,
    "border_count": 128,
}
FIXED_LIGHTGBM_PARAMS = {
    "max_depth": 8,
    "num_leaves": 30,
    "n_estimators": 3800,
    "learning_rate": 0.015831790730798656,
    "min_child_samples": 280,
    "reg_lambda": 0.15805460542063748,
    "reg_alpha": 3.920649538192273e-05,
    "colsample_bytree": 0.6054519676764284,
    "subsample": 0.9935009272600186,
    "min_split_gain": 0.9053872910470832,
    "max_bin": 1023,
    "cat_smooth": 61.33613152281461,
    "cat_l2": 2.653555349507821,
}
FIXED_XGBOOST_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03162931523007232,
    "max_depth": 4,
    "min_child_weight": 43.61361810092031,
    "gamma": 4.251923725207756,
    "subsample": 0.6324027744325311,
    "colsample_bytree": 0.5996292985062543,
    "reg_lambda": 0.2204066172541325,
    "reg_alpha": 7.878776514694756,
    "max_bin": 1024,
    "max_cat_to_onehot": 9,
}

# RealMLP base settings; Optuna replaces the architecture and optimization fields per fold.
REALMLP_PARAMS = {
    "device": "cuda:0",
    "random_state": RANDOM_SEED,
    "n_cv": 1,
    "n_refit": 0,
    "n_threads": CPU_THREADS,
    "n_epochs": 256,
    "batch_size": 512,
    "predict_batch_size": 2048,
    "hidden_sizes": [256, 256, 256],
    "lr": 0.03,
    "p_drop": 0.15,
    "wd": 0.01,
    "val_metric_name": "cross_entropy",
    "use_ls": False,
    "verbosity": 1,
}

ENSEMBLE_WEIGHTS = {
    "realmlp": 0.25,
    "catboost": 0.25,
    "lightgbm": 0.25,
    "xgboost": 0.25,
}

MLP_RAW_CATEGORICAL_COLUMNS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]
MLP_CATEGORICAL_COLUMNS = MLP_RAW_CATEGORICAL_COLUMNS + [
    LOAN_STRUCTURE_FEATURE,
    CAREER_PROFILE_FEATURE,
]
MLP_ENGINEERED_FEATURES = [
    "CREDIT_TERM",
    "INCOME_TO_REGION_MEAN",
    "OWN_CAR_AGE_CLEAN",
    "DIFF_RATIO",
    "NET_INCOME_PER_MEMBER",
    "CAR_FRESHNESS",
    "GOODS_PRICE_TO_ANNUITY",
    NEW_INTERACTION_FEATURE,
    LOAN_STRUCTURE_FEATURE,
    CAREER_PROFILE_FEATURE,
]
TREE_ENGINEERED_FEATURES = [
    "CREDIT_TERM",
    "INCOME_TO_REGION_MEAN",
    "OWN_CAR_AGE_CLEAN",
    "DIFF_RATIO",
    "NET_INCOME_PER_MEMBER",
    "CAR_FRESHNESS",
    "GOODS_PRICE_TO_ANNUITY",
    NEW_INTERACTION_FEATURE,
    LOAN_STRUCTURE_FEATURE,
    CAREER_PROFILE_FEATURE,
]

# Hybrid GMM loan-structure segmentation settings copied from your prior script.
LOAN_STRUCTURE_TERM_PRECISION = 2
LOAN_STRUCTURE_GOODS_PRECISION = 3
LOAN_STRUCTURE_SPIKE_MIN_SHARE = 0.03
LOAN_STRUCTURE_SPIKE_DOMINANCE = 2.0
GMM_MAX_COMPONENTS = 30
GMM_N_INIT = 3
GMM_MAX_ITER = 300
GMM_REG_COVAR = 1e-5
GMM_MIN_COMPONENT_ROWS = 100
GMM_RANDOM_STATE = RANDOM_SEED
GMM_STABILITY_REPEATS = 3
TERM_SPIKE_ONLY_REVIEW_SUPPORT = 500

# Keep schema version 1 to permit reuse of the matching legacy GMM cache.
# Increase this only if the GMM fitting/transform algorithm or cache payload changes.
GMM_CACHE_SCHEMA_VERSION = 1
GMM_CACHE_INPUT_COLUMNS = [
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
]

TREE_MODEL_NAMES = ["catboost", "lightgbm", "xgboost"]
ALL_MODEL_NAMES = ["realmlp", "catboost", "lightgbm", "xgboost"]


# =============================================================================
# 3. ENVIRONMENT / ATOMIC CHECKPOINT HELPERS
# =============================================================================

def get_package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def print_section(title: str) -> None:
    print("\n" + "=" * 105, flush=True)
    print(title, flush=True)
    print("=" * 105, flush=True)


def _tmp_path(path: Path) -> Path:
    return path.with_name(path.stem + ".tmp" + path.suffix)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _tmp_path(path)
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _tmp_path(path)
    with temporary.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _tmp_path(path)
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _tmp_path(path)
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_model_save(model: Any, path: Path, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _tmp_path(path)
    if kind == "catboost":
        model.save_model(str(temporary), format="cbm")
    elif kind == "lightgbm":
        model.booster_.save_model(str(temporary))
    elif kind == "xgboost":
        model.save_model(str(temporary))
    elif kind == "pickle":
        with temporary.open("wb") as file:
            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
    else:
        raise ValueError(f"Unknown model serialization kind: {kind}")
    os.replace(temporary, path)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_is_complete(
    marker_path: Path,
    expected_stage: str,
    config_hash: str,
) -> bool:
    if not marker_path.exists():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        return (
            marker.get("stage") == expected_stage
            and marker.get("config_sha256") == config_hash
        )
    except Exception:
        return False


def mark_complete(
    marker_path: Path,
    stage: str,
    config_hash: str,
    artifacts: list[Path],
) -> None:
    missing = [str(path) for path in artifacts if not path.exists()]
    if missing:
        raise RuntimeError(f"Cannot mark {stage} complete; missing artifacts: {missing}")
    atomic_json(
        marker_path,
        {
            "stage": stage,
            "config_sha256": config_hash,
            "completed_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "artifacts": [
                {"path": str(path), "size_bytes": int(path.stat().st_size)}
                for path in artifacts
            ],
        },
    )


def check_probabilities(name: str, probabilities: np.ndarray) -> None:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError(f"{name}: predictions must be finite and one-dimensional")
    if ((p < 0.0) | (p > 1.0)).any():
        raise ValueError(f"{name}: predictions must be in [0, 1]")


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    check_probabilities("Metric calculation", p)
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "average_precision": float(average_precision_score(y_true, p)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, p)),
    }


def positive_class_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X))
    if probabilities.ndim != 2:
        raise RuntimeError("predict_proba must return a 2D array")
    classes = np.asarray(getattr(model, "classes_", np.array([0, 1])))
    positive_indices = np.where(classes == 1)[0]
    if len(positive_indices) == 1:
        return probabilities[:, int(positive_indices[0])].astype(np.float64)
    if probabilities.shape[1] == 2:
        return probabilities[:, 1].astype(np.float64)
    raise RuntimeError("Could not identify positive-class probability")


# Local CUDA validation: RealMLP, CatBoost and XGBoost require GPU 0.
def detect_nvidia_gpu() -> bool:
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, check=False)
        return result.returncode == 0
    except (FileNotFoundError, OSError):
        return False

if not detect_nvidia_gpu():
    raise RuntimeError("nvidia-smi did not detect an NVIDIA GPU. Check the driver and CUDA installation.")
if not torch.cuda.is_available():
    raise RuntimeError(
        f"CUDA-enabled PyTorch is required for RealMLP; torch={torch.__version__}, "
        f"torch.version.cuda={torch.version.cuda}"
    )
if Version(xgb.__version__.split("+")[0]) < Version("2.0.0"):
    raise RuntimeError(f"XGBoost >= 2.0.0 is required for CUDA; found {xgb.__version__}")
try:
    torch.cuda.set_device(0)
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(max(1, min(2, CPU_THREADS)))
except RuntimeError:
    pass
DEVICE = "cuda:0"
REALMLP_PARAMS["device"] = DEVICE

print_section("HOME CREDIT — FOUR-MODEL OPTUNA GPU ENSEMBLE WITH FOLD-LOCAL EXT_SOURCE IMPUTATION")
print(f"Python       : {sys.version.split()[0]}")
print(f"PyTorch      : {torch.__version__}")
print(f"PyTorch CUDA : {torch.version.cuda}")
print(f"CUDA GPU     : {torch.cuda.get_device_name(0)}")
print(f"GPU memory   : {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GiB")
print(f"CatBoost     : {get_package_version('catboost')} (GPU)")
print(f"LightGBM     : {get_package_version('lightgbm')} (CPU)")
print(f"XGBoost      : {xgb.__version__} (CUDA)")
print(f"PyTabKit     : {get_package_version('pytabkit')} (RealMLP CUDA)")
print(f"CPU threads  : {CPU_THREADS}")
print(f"Output       : {OUTPUT_DIR}")
print(f"GMM cache    : {GMM_CACHE_DIR}")
print(f"Imputation   : {IMPUTATION_EXPERIMENT_DIR}")


# =============================================================================
# 4. LOAD / VALIDATE INPUTS AND FIXED FOLDS
# =============================================================================

for required_path in [
    TRAIN_FILE,
    TEST_FILE,
    IMPUTATION_MODELS_DIR,
    IMPUTATION_CLASSIFIER_SCENARIO_DIR,
]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required path does not exist: {required_path}")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

if ID_COL not in train.columns or ID_COL not in test.columns or TARGET_COL not in train.columns:
    raise KeyError("Train/test must contain SK_ID_CURR; train must also contain TARGET")
if train[ID_COL].isna().any() or train[ID_COL].duplicated().any():
    raise ValueError("Train IDs must be unique and non-missing")
if test[ID_COL].isna().any() or test[ID_COL].duplicated().any():
    raise ValueError("Test IDs must be unique and non-missing")
if train[TARGET_COL].isna().any() or set(train[TARGET_COL].unique()) != {0, 1}:
    raise ValueError("TARGET must be non-missing and contain exactly {0, 1}")
if set(train[ID_COL]).intersection(set(test[ID_COL])):
    raise ValueError("Train/test IDs overlap")

if SAMPLE_SUB_FILE.exists():
    sample_sub = pd.read_csv(SAMPLE_SUB_FILE)
    if ID_COL not in sample_sub.columns or TARGET_COL not in sample_sub.columns:
        raise ValueError("sample_submission.csv must contain SK_ID_CURR and TARGET")
    if set(sample_sub[ID_COL]) != set(test[ID_COL]):
        raise ValueError("sample_submission IDs do not match test IDs")
else:
    sample_sub = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), TARGET_COL: np.nan})

feature_columns = [
    column for column in train.columns if column not in [ID_COL, TARGET_COL]
]
test_feature_columns = [column for column in test.columns if column != ID_COL]
if set(feature_columns) != set(test_feature_columns):
    raise ValueError(
        "Train/test feature mismatch: "
        f"missing={sorted(set(feature_columns) - set(test_feature_columns))}; "
        f"extra={sorted(set(test_feature_columns) - set(feature_columns))}"
    )

required_columns = [
    *MLP_RAW_CATEGORICAL_COLUMNS,
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "AMT_INCOME_TOTAL",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "OWN_CAR_AGE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_ID_PUBLISH",
    *EXT_SOURCE_COLUMNS,
]
missing_required = sorted(set(required_columns) - set(feature_columns))
if missing_required:
    raise KeyError(f"Missing required predictors: {missing_required}")

# GOODS_PRICE_TO_ANNUITY must not be an original classifier predictor.
if "GOODS_PRICE_TO_ANNUITY" in feature_columns:
    raise ValueError(
        "GOODS_PRICE_TO_ANNUITY unexpectedly exists in the raw classifier data. "
        "This script allows it only as a temporary saved-imputer input feature."
    )

reserved_generated = set(
    MLP_ENGINEERED_FEATURES
    + [LOAN_STRUCTURE_FEATURE]
    + REQUESTED_SHARED_FEATURES
    + EXT_SOURCE_MISSING_COLUMNS
)
engineered_collisions = sorted(reserved_generated.intersection(feature_columns))
if engineered_collisions:
    raise ValueError(f"Raw data contains reserved generated columns: {engineered_collisions}")

X_train_original = train.loc[:, feature_columns].copy()
X_test_original = test.loc[:, feature_columns].copy()
y = train[TARGET_COL].astype(np.int32).to_numpy()
train_ids = train[ID_COL].to_numpy()
test_ids = test[ID_COL].to_numpy()
n_train, n_test = len(train), len(test)

# Keep the original eight categorical columns categorical even if loaded as numbers.
raw_categorical_columns = [
    column
    for column in feature_columns
    if (
        pd.api.types.is_object_dtype(X_train_original[column].dtype)
        or pd.api.types.is_string_dtype(X_train_original[column].dtype)
        or isinstance(X_train_original[column].dtype, pd.CategoricalDtype)
    )
]
for column in MLP_RAW_CATEGORICAL_COLUMNS:
    if column not in raw_categorical_columns:
        raw_categorical_columns.append(column)
    X_train_original[column] = X_train_original[column].astype("string")
    X_test_original[column] = X_test_original[column].astype("string")

for column in feature_columns:
    if column not in raw_categorical_columns:
        X_train_original[column] = pd.to_numeric(X_train_original[column], errors="coerce")
        X_test_original[column] = pd.to_numeric(X_test_original[column], errors="coerce")
        X_train_original[column] = X_train_original[column].replace([np.inf, -np.inf], np.nan)
        X_test_original[column] = X_test_original[column].replace([np.inf, -np.inf], np.nan)

for column in raw_categorical_columns:
    X_train_original[column] = X_train_original[column].astype("string")
    X_test_original[column] = X_test_original[column].astype("string")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
fold_splits = list(skf.split(X_train_original, y))


def load_fold_indices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        train_keys = [
            key for key in keys
            if ("train" in key.lower() or "fit" in key.lower())
            and "valid" not in key.lower()
        ]
        valid_keys = [key for key in keys if "valid" in key.lower()]
        if not train_keys or not valid_keys:
            raise KeyError(f"Could not identify train/valid arrays in {path}; keys={keys}")
        return (
            np.asarray(archive[train_keys[0]], dtype=np.int64),
            np.asarray(archive[valid_keys[0]], dtype=np.int64),
        )


print_section("VERIFYING SAVED IMPUTATION FOLDS AND ARTIFACTS")
print(f"Train rows: {n_train:,} | Test rows: {n_test:,} | Predictors: {len(feature_columns)}")
print(f"Positive rows: {int(y.sum()):,} | Positive rate: {y.mean():.8%}")
for fold_number, (fit_indices, valid_indices) in enumerate(fold_splits, start=1):
    saved_indices_path = (
        IMPUTATION_CLASSIFIER_SCENARIO_DIR
        / "CHECKPOINTS"
        / "FOLDS"
        / f"FOLD_{fold_number:02d}"
        / "TRAIN_VALID_INDICES.npz"
    )
    if not saved_indices_path.exists():
        raise FileNotFoundError(f"Saved imputation fold indices not found: {saved_indices_path}")
    saved_fit, saved_valid = load_fold_indices(saved_indices_path)
    if not np.array_equal(np.sort(fit_indices), np.sort(saved_fit)):
        raise RuntimeError(f"Fold {fold_number}: training indices differ from imputation experiment")
    if not np.array_equal(np.sort(valid_indices), np.sort(saved_valid)):
        raise RuntimeError(f"Fold {fold_number}: validation indices differ from imputation experiment")

    fold_values_path = IMPUTATION_MODELS_DIR / f"FOLD_{fold_number:02d}" / "IMPUTED_VALUES.npz"
    if not fold_values_path.exists():
        raise FileNotFoundError(f"Saved training imputed values not found: {fold_values_path}")
    for source in EXT_SOURCE_COLUMNS:
        model_path = (
            IMPUTATION_MODELS_DIR
            / f"FOLD_{fold_number:02d}"
            / source
            / "imputer.cbm"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Saved {source} imputer not found: {model_path}")
    print(f"Fold {fold_number}: exact split verified; 3 imputers and saved values found.")


# Include all saved imputer artifacts in the experiment signature.
imputation_signatures = {}
for fold_number in range(1, N_SPLITS + 1):
    values_path = IMPUTATION_MODELS_DIR / f"FOLD_{fold_number:02d}" / "IMPUTED_VALUES.npz"
    imputation_signatures[f"fold_{fold_number}_values"] = file_signature(values_path)
    for source in EXT_SOURCE_COLUMNS:
        model_path = (
            IMPUTATION_MODELS_DIR
            / f"FOLD_{fold_number:02d}"
            / source
            / "imputer.cbm"
        )
        imputation_signatures[f"fold_{fold_number}_{source}_model"] = file_signature(model_path)
    index_path = (
        IMPUTATION_CLASSIFIER_SCENARIO_DIR
        / "CHECKPOINTS"
        / "FOLDS"
        / f"FOLD_{fold_number:02d}"
        / "TRAIN_VALID_INDICES.npz"
    )
    imputation_signatures[f"fold_{fold_number}_indices"] = file_signature(index_path)

config_payload = {
    "experiment": "HOME_CREDIT_FOUR_MODEL_OPTUNA_GPU_5FOLD_IMPUTED_EXT_SOURCE_1_2_3_PLUS_9_SHARED_GMM",
    "train_file": file_signature(TRAIN_FILE),
    "test_file": file_signature(TEST_FILE),
    "sample_submission_file": file_signature(SAMPLE_SUB_FILE) if SAMPLE_SUB_FILE.exists() else None,
    "imputation_experiment_dir": str(IMPUTATION_EXPERIMENT_DIR),
    "imputation_artifacts": imputation_signatures,
    "train_shape": list(train.shape),
    "test_shape": list(test.shape),
    "folds": {
        "method": "StratifiedKFold",
        "n_splits": N_SPLITS,
        "shuffle": True,
        "random_state": RANDOM_SEED,
    },
    "cpu_threads": CPU_THREADS,
    "devices": {"realmlp": "cuda:0", "catboost": "GPU 0", "lightgbm": "CPU", "xgboost": "cuda:0"},
    "training_method": "Optuna with inner StratifiedKFold(5) in each outer fold",
    "tree_params": {
        "catboost": FIXED_CATBOOST_PARAMS,
        "lightgbm": FIXED_LIGHTGBM_PARAMS,
        "xgboost": FIXED_XGBOOST_PARAMS,
    },
    "realmlp_params": REALMLP_PARAMS,
    "ensemble_weights": ENSEMBLE_WEIGHTS,
    "ext_source_scenario": "IMPUTE_EXT_SOURCE_1_2_3 with three original missing flags",
    "shared_features": REQUESTED_SHARED_FEATURES,
    "additional_engineered_features": {
        "DIFF_RATIO": "(AMT_CREDIT - AMT_GOODS_PRICE) / AMT_GOODS_PRICE; signed, not squared",
        "NET_INCOME_PER_MEMBER": "(AMT_INCOME_TOTAL - AMT_ANNUITY) / CNT_FAM_MEMBERS",
        "CAR_FRESHNESS": "1 / (1 + clip(OWN_CAR_AGE, lower=0)); preserve missing values",
    },
    "career_profile": {
        "feature": CAREER_PROFILE_FEATURE,
        "minimum_training_count": CAREER_PROFILE_MIN_COUNT,
        "pensioner_override": CAREER_PROFILE_PENSIONER_VALUE,
        "fallback_order": [
            "INCOME+OCCUPATION+ORGANIZATION",
            "INCOME+OCCUPATION",
            "INCOME+ORGANIZATION",
            "INCOME",
        ],
        "fit_scope": "current training partition only; refit within every Optuna inner fold and RealMLP epoch-selection split",
        "raw_categorical_features_retained": True,
    },
    "goods_price_to_annuity": {
        "classifier_feature": False,
        "saved_imputer_input_feature": True,
    },
    "gmm_cache": {
        "directory": str(GMM_CACHE_DIR),
        "schema_version": GMM_CACHE_SCHEMA_VERSION,
        "input_columns": GMM_CACHE_INPUT_COLUMNS,
    },
    "neighbor_count": N_NEIGHBORS,
    "gmm": {
        "max_components": GMM_MAX_COMPONENTS,
        "n_init": GMM_N_INIT,
        "max_iter": GMM_MAX_ITER,
        "reg_covar": GMM_REG_COVAR,
        "min_component_rows": GMM_MIN_COMPONENT_ROWS,
        "stability_repeats": GMM_STABILITY_REPEATS,
    },
}
config_payload["pipeline_version"] = "2026-11-15-optuna-gpu-inner5fold-career-profile-v2"
CONFIG_HASH = sha256_json(config_payload)
RUN_CONFIG_PATH = OUTPUT_DIR / "run_config.json"
if RUN_CONFIG_PATH.exists():
    old_config = json.loads(RUN_CONFIG_PATH.read_text(encoding="utf-8"))
    if old_config.get("config_sha256") != CONFIG_HASH:
        previous_path = OUTPUT_DIR / "run_config_previous.json"
        if not previous_path.exists():
            atomic_json(previous_path, old_config)
        print(
            "Existing run configuration differs from this GPU/Optuna run. "
            "The new signature will force feature-cache validation/rebuild and prevent "
            "old fixed-model checkpoints from being reused.",
            flush=True,
        )
        atomic_json(RUN_CONFIG_PATH, {"config_sha256": CONFIG_HASH, "configuration": config_payload})
else:
    atomic_json(RUN_CONFIG_PATH, {"config_sha256": CONFIG_HASH, "configuration": config_payload})


# =============================================================================
# 5. REBUILD THE SAVED IMPUTER INPUT SCHEMA — INCLUDING GOODS_PRICE_TO_ANNUITY
# =============================================================================


def make_ratio_features_for_imputer(frame: pd.DataFrame) -> pd.DataFrame:
    """Temporary imputer-only features; never used as classifier features."""
    output = frame.copy()
    annuity = pd.to_numeric(output["AMT_ANNUITY"], errors="coerce").replace(0, np.nan)
    credit = pd.to_numeric(output["AMT_CREDIT"], errors="coerce")
    goods = pd.to_numeric(output["AMT_GOODS_PRICE"], errors="coerce")
    output["CREDIT_TERM"] = (credit / annuity).replace([np.inf, -np.inf], np.nan)
    # Deliberately retained ONLY because the stored imputers were trained with it.
    output["GOODS_PRICE_TO_ANNUITY"] = (goods / annuity).replace(
        [np.inf, -np.inf], np.nan
    )
    return output


def build_imputer_input_frame(
    raw_frame: pd.DataFrame,
    target_source: str,
    feature_names: list[str],
    categorical_indices: list[int],
) -> pd.DataFrame:
    """Recreate the exact stored CatBoost imputer feature names and column order."""
    all_features = make_ratio_features_for_imputer(raw_frame)
    for source in EXT_SOURCE_COLUMNS:
        if source != target_source:
            all_features[f"{source}_MISSING"] = (
                pd.to_numeric(raw_frame[source], errors="coerce").isna().astype(np.int8)
            )
    missing_features = [name for name in feature_names if name not in all_features.columns]
    if missing_features:
        raise KeyError(
            f"Cannot reconstruct {target_source} imputer inputs; missing={missing_features}"
        )
    if target_source in feature_names:
        raise RuntimeError(f"{target_source} must not be an input to its own imputer")
    if "GOODS_PRICE_TO_ANNUITY" not in feature_names:
        raise RuntimeError(
            f"Stored {target_source} imputer does not expose GOODS_PRICE_TO_ANNUITY. "
            "Verify that the correct saved imputation experiment was selected."
        )

    result = all_features.loc[:, feature_names].copy()
    categorical_names = {feature_names[index] for index in categorical_indices}
    for column in feature_names:
        if column in categorical_names:
            result[column] = (
                result[column].astype("string").fillna(MISSING_CATEGORY).astype(str)
            )
        else:
            result[column] = pd.to_numeric(result[column], errors="coerce")
            result[column] = result[column].replace([np.inf, -np.inf], np.nan)
    return result


def load_and_validate_training_imputations(fold_number: int) -> dict[str, np.ndarray]:
    path = IMPUTATION_MODELS_DIR / f"FOLD_{fold_number:02d}" / "IMPUTED_VALUES.npz"
    with np.load(path, allow_pickle=False) as archive:
        result = {
            source: np.asarray(archive[source], dtype=np.float64)
            for source in EXT_SOURCE_COLUMNS
        }
    for source in EXT_SOURCE_COLUMNS:
        raw = pd.to_numeric(X_train_original[source], errors="coerce").to_numpy(
            dtype=np.float64
        )
        imputed = result[source]
        if imputed.shape != (n_train,):
            raise ValueError(f"Fold {fold_number} {source}: unexpected shape {imputed.shape}")
        if not np.isfinite(imputed).all():
            raise ValueError(f"Fold {fold_number} {source}: saved values contain NaN/inf")
        observed = np.isfinite(raw)
        if not np.allclose(imputed[observed], raw[observed], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Fold {fold_number} {source}: observed values were changed")
        if ((imputed < 0.0) | (imputed > 1.0)).any():
            raise ValueError(f"Fold {fold_number} {source}: values outside [0, 1]")
    return result


def generate_test_imputations(
    fold_number: int,
    fit_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Impute test with the matching fold's saved imputers and fold-train clipping."""
    fold_dir = IMPUTATION_TEST_DIR / f"FOLD_{fold_number:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    values_path = fold_dir / "TEST_IMPUTED_VALUES.npz"
    audit_path = fold_dir / "TEST_IMPUTATION_AUDIT.csv"
    marker_path = fold_dir / "IMPUTATION_TEST_COMPLETE.json"

    if checkpoint_is_complete(marker_path, "test_imputation", CONFIG_HASH):
        if not values_path.exists() or not audit_path.exists():
            raise RuntimeError(f"Fold {fold_number}: test-imputation checkpoint is incomplete")
        with np.load(values_path, allow_pickle=False) as archive:
            result = {
                source: np.asarray(archive[source], dtype=np.float64)
                for source in EXT_SOURCE_COLUMNS
            }
        for source in EXT_SOURCE_COLUMNS:
            raw = pd.to_numeric(X_test_original[source], errors="coerce").to_numpy(
                dtype=np.float64
            )
            values = result[source]
            if values.shape != (n_test,) or not np.isfinite(values).all():
                raise RuntimeError(f"Fold {fold_number} {source}: cached test values invalid")
            observed = np.isfinite(raw)
            if not np.allclose(values[observed], raw[observed], rtol=0.0, atol=1e-12):
                raise RuntimeError(f"Fold {fold_number} {source}: cached observed values changed")
        print(f"Fold {fold_number}: restored completed test imputations.", flush=True)
        return result

    print_section(f"GENERATING TEST EXT_SOURCE IMPUTATIONS — FOLD {fold_number}/{N_SPLITS}")
    output: dict[str, np.ndarray] = {}
    audit_rows = []

    for source in EXT_SOURCE_COLUMNS:
        model_path = (
            IMPUTATION_MODELS_DIR
            / f"FOLD_{fold_number:02d}"
            / source
            / "imputer.cbm"
        )
        imputer = CatBoostRegressor()
        imputer.load_model(str(model_path))
        feature_names = list(imputer.feature_names_)
        categorical_indices = list(imputer.get_cat_feature_indices())
        if len(feature_names) != 35:
            raise RuntimeError(
                f"Fold {fold_number} {source}: expected 35 saved imputer inputs, "
                f"found {len(feature_names)}"
            )

        X_imp_test = build_imputer_input_frame(
            X_test_original,
            source,
            feature_names,
            categorical_indices,
        )
        raw = pd.to_numeric(X_test_original[source], errors="coerce").to_numpy(
            dtype=np.float64
        )
        missing = ~np.isfinite(raw)
        result = raw.copy()

        fold_train_source = pd.to_numeric(
            X_train_original.iloc[fit_indices][source], errors="coerce"
        ).to_numpy(dtype=np.float64)
        known_fold_train = fold_train_source[np.isfinite(fold_train_source)]
        if known_fold_train.size == 0:
            raise RuntimeError(f"Fold {fold_number} {source}: no observed fold-training values")
        clip_min = float(known_fold_train.min())
        clip_max = float(known_fold_train.max())

        clipped_count = 0
        if missing.any():
            prediction = np.asarray(
                imputer.predict(X_imp_test.loc[missing]), dtype=np.float64
            ).reshape(-1)
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"Fold {fold_number} {source}: imputer produced non-finite values")
            clipped = np.clip(prediction, clip_min, clip_max)
            clipped_count = int(np.sum(clipped != prediction))
            result[missing] = clipped

        if not np.isfinite(result).all():
            raise RuntimeError(f"Fold {fold_number} {source}: test imputation is incomplete")
        output[source] = result
        audit_rows.append(
            {
                "FOLD": fold_number,
                "SOURCE": source,
                "TEST_ROWS": n_test,
                "OBSERVED_TEST_VALUES": int((~missing).sum()),
                "MISSING_TEST_VALUES_IMPUTED": int(missing.sum()),
                "CLIPPED_PREDICTIONS": clipped_count,
                "CLIP_MIN_FROM_FOLD_TRAIN": clip_min,
                "CLIP_MAX_FROM_FOLD_TRAIN": clip_max,
                "IMPUTED_MIN": float(result.min()),
                "IMPUTED_MAX": float(result.max()),
                "MODEL_FEATURE_COUNT": len(feature_names),
                "GOODS_PRICE_TO_ANNUITY_IN_IMPUTER": (
                    "GOODS_PRICE_TO_ANNUITY" in feature_names
                ),
                "MODEL_CATEGORICAL_FEATURES": [
                    feature_names[index] for index in categorical_indices
                ],
            }
        )
        del imputer, X_imp_test
        gc.collect()

    atomic_npz(values_path, **output)
    atomic_csv(audit_path, pd.DataFrame(audit_rows))
    mark_complete(
        marker_path,
        "test_imputation",
        CONFIG_HASH,
        [values_path, audit_path],
    )
    print(pd.DataFrame(audit_rows).to_string(index=False), flush=True)
    return output


# =============================================================================
# 6. ADD THE WINNING EXT_SOURCE IMPUTATION + NINE SHARED FEATURES
# =============================================================================


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator != 0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def augment_with_ext_source_scenario(
    original_frame: pd.DataFrame,
    imputed_values: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Insert three imputed sources + original flags, then add nine shared features."""
    output = original_frame.copy()
    for source in EXT_SOURCE_COLUMNS:
        raw = pd.to_numeric(original_frame[source], errors="coerce").to_numpy(
            dtype=np.float64
        )
        imputed = np.asarray(imputed_values[source], dtype=np.float64)
        if imputed.shape != (len(original_frame),) or not np.isfinite(imputed).all():
            raise ValueError(f"Invalid imputed vector for {source}")
        missing = ~np.isfinite(raw)
        final_values = raw.copy()
        final_values[missing] = imputed[missing]
        if not np.isfinite(final_values).all():
            raise RuntimeError(f"{source}: missing values remain after imputation")
        output[source] = final_values.astype(np.float32)
        output[f"{source}_MISSING"] = missing.astype(np.int8)

    ext_matrix = output.loc[:, EXT_SOURCE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(ext_matrix).all():
        raise RuntimeError("Cannot calculate EXT_SOURCE summaries before full imputation")
    output["EXT_SOURCE_MEAN"] = ext_matrix.mean(axis=1).astype(np.float32)
    output["EXT_SOURCE_STD"] = ext_matrix.std(axis=1, ddof=0).astype(np.float32)
    output["EXT_SOURCE_MIN"] = ext_matrix.min(axis=1).astype(np.float32)
    output["EXT_SOURCE_MAX"] = ext_matrix.max(axis=1).astype(np.float32)
    output["EXT_SOURCE_PRODUCT"] = ext_matrix.prod(axis=1).astype(np.float32)

    income = pd.to_numeric(output["AMT_INCOME_TOTAL"], errors="coerce")
    annuity = pd.to_numeric(output["AMT_ANNUITY"], errors="coerce")
    credit = pd.to_numeric(output["AMT_CREDIT"], errors="coerce")
    days_birth = pd.to_numeric(output["DAYS_BIRTH"], errors="coerce")
    days_employed_clean = pd.to_numeric(output["DAYS_EMPLOYED"], errors="coerce").replace(
        365243, np.nan
    )
    days_id_publish = pd.to_numeric(output["DAYS_ID_PUBLISH"], errors="coerce")

    output["ANNUITY_TO_INCOME"] = safe_ratio(annuity, income).astype(np.float32)
    output["CREDIT_TO_INCOME"] = safe_ratio(credit, income).astype(np.float32)
    output["EMPLOYED_TO_AGE"] = safe_ratio(
        days_employed_clean, days_birth
    ).astype(np.float32)
    output["DAYS_ID_TO_BIRTH"] = safe_ratio(
        days_id_publish, days_birth
    ).astype(np.float32)

    missing_shared = [column for column in REQUESTED_SHARED_FEATURES if column not in output]
    if missing_shared:
        raise RuntimeError(f"Requested shared features were not generated: {missing_shared}")
    if "GOODS_PRICE_TO_ANNUITY" in output.columns:
        raise RuntimeError(
            "GOODS_PRICE_TO_ANNUITY leaked into the classifier frame; it must remain imputer-only"
        )
    return output


# =============================================================================
# 7. HYBRID GAUSSIAN LOAN-STRUCTURE SEGMENTER — LEGACY METHOD
# =============================================================================

class HybridLoanStructureSegmenter:
    """Fold-local predictor-only spike handling + BIC-selected Gaussian mixtures."""

    def __init__(self, fold_number: int):
        self.fold_number = int(fold_number)
        self.fitted = False

    @staticmethod
    def calculate_ratios(frame: pd.DataFrame):
        credit = pd.to_numeric(frame["AMT_CREDIT"], errors="coerce")
        annuity = pd.to_numeric(frame["AMT_ANNUITY"], errors="coerce")
        goods_price = pd.to_numeric(frame["AMT_GOODS_PRICE"], errors="coerce")
        annuity = annuity.replace([np.inf, -np.inf], np.nan).where(annuity > 0)
        goods_price = goods_price.replace([np.inf, -np.inf], np.nan).where(goods_price > 0)
        term = (credit / annuity).replace([np.inf, -np.inf], np.nan).where(lambda z: z > 0)
        goods = (credit / goods_price).replace([np.inf, -np.inf], np.nan).where(lambda z: z > 0)
        return term, goods

    @staticmethod
    def detect_spike(values: np.ndarray, precision: int, axis_name: str) -> dict[str, Any]:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size == 0:
            return {
                "value": None,
                "share": 0.0,
                "rounded_top_count": 0,
                "second_count": 0,
                "valid_count": 0,
                "detected": False,
                "axis": axis_name,
                "precision": int(precision),
            }
        counts = pd.Series(np.round(values, precision)).value_counts()
        top_value = float(counts.index[0])
        top_count = int(counts.iloc[0])
        second_count = int(counts.iloc[1]) if len(counts) > 1 else 0
        share = top_count / len(values)
        detected = bool(
            len(counts) >= 2
            and share >= LOAN_STRUCTURE_SPIKE_MIN_SHARE
            and top_count >= LOAN_STRUCTURE_SPIKE_DOMINANCE * second_count
        )
        return {
            "value": top_value if detected else None,
            "share": float(share if detected else 0.0),
            "rounded_top_value": top_value,
            "rounded_top_count": top_count,
            "second_count": second_count,
            "valid_count": int(len(values)),
            "detected": detected,
            "axis": axis_name,
            "precision": int(precision),
        }

    @staticmethod
    def spike_mask(values: np.ndarray, spike_info: dict[str, Any], precision: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(values) & (values > 0)
        value = spike_info["value"]
        if value is None:
            return np.zeros(len(values), dtype=bool)
        return valid & (np.round(values, precision) == value)

    def fit_gmm_by_bic(self, X_raw: np.ndarray, branch_name: str):
        X_raw = np.asarray(X_raw, dtype=np.float64)
        if X_raw.ndim == 1:
            X_raw = X_raw.reshape(-1, 1)
        if X_raw.ndim != 2 or X_raw.shape[1] not in (1, 2):
            raise ValueError(f"{branch_name}: expected 1D or 2D GMM input")
        if not np.isfinite(X_raw).all():
            raise ValueError(f"{branch_name}: non-finite GMM input")
        if len(X_raw) < GMM_MIN_COMPONENT_ROWS:
            raise ValueError(f"{branch_name}: insufficient training rows: {len(X_raw)}")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        max_k = min(GMM_MAX_COMPONENTS, max(1, len(X_scaled) // GMM_MIN_COMPONENT_ROWS))
        candidate_rows = []
        candidate_sizes = []
        best_model, best_bic, best_counts, best_k = None, np.inf, None, None

        for k in range(1, max_k + 1):
            model = GaussianMixture(
                n_components=k,
                covariance_type="full",
                reg_covar=GMM_REG_COVAR,
                max_iter=GMM_MAX_ITER,
                n_init=GMM_N_INIT,
                init_params="kmeans",
                random_state=GMM_RANDOM_STATE + 100 * self.fold_number + k,
            )
            model.fit(X_scaled)
            assignments = model.predict(X_scaled)
            counts = np.bincount(assignments, minlength=k)
            bic = float(model.bic(X_scaled))
            min_count = int(counts.min())
            eligible = bool(model.converged_ and min_count >= GMM_MIN_COMPONENT_ROWS)
            means_original = scaler.inverse_transform(model.means_)
            order = (
                np.argsort(means_original[:, 0])
                if means_original.shape[1] == 1
                else np.lexsort((means_original[:, 1], means_original[:, 0]))
            )
            candidate_rows.append(
                {
                    "branch": branch_name,
                    "n_components": int(k),
                    "bic": bic,
                    "converged": bool(model.converged_),
                    "minimum_hard_cluster_rows": min_count,
                    "maximum_hard_cluster_rows": int(counts.max()),
                    "eligible_for_selection": eligible,
                    "is_search_boundary": bool(k == max_k),
                    "iterations": int(model.n_iter_),
                }
            )
            for rank, old in enumerate(order, start=1):
                center = means_original[old]
                row = {
                    "branch": branch_name,
                    "n_components": int(k),
                    "component_rank_by_center": int(rank),
                    "original_model_component": int(old),
                    "hard_count": int(counts[old]),
                    "mixture_weight": float(model.weights_[old]),
                    "weight_expected_rows": float(model.weights_[old] * len(X_scaled)),
                    "mean_log_feature_1": float(center[0]),
                    "selected_by_bic": False,
                }
                if len(center) == 1:
                    row["geometric_center_credit_term"] = float(np.exp(center[0]))
                    row["geometric_center_credit_to_goods"] = np.nan
                else:
                    row["mean_log_feature_2"] = float(center[1])
                    row["geometric_center_credit_term"] = float(np.exp(center[0]))
                    row["geometric_center_credit_to_goods"] = float(np.exp(center[1]))
                candidate_sizes.append(row)
            if eligible and bic < best_bic:
                best_model, best_bic, best_counts, best_k = model, bic, counts.copy(), k

        if best_model is None:
            raise RuntimeError(
                f"{branch_name}: no converged GMM met minimum hard-cluster support "
                f"{GMM_MIN_COMPONENT_ROWS}"
            )

        for row in candidate_sizes:
            row["selected_by_bic"] = int(row["n_components"]) == int(best_k)

        means = scaler.inverse_transform(best_model.means_)
        order = (
            np.argsort(means[:, 0])
            if means.shape[1] == 1
            else np.lexsort((means[:, 1], means[:, 0]))
        )
        rank_map = np.empty(best_model.n_components, dtype=np.int32)
        components = []
        for rank, old in enumerate(order, start=1):
            rank_map[int(old)] = rank
            center = means[old]
            row = {
                "ranked_component": int(rank),
                "original_model_component": int(old),
                "weight": float(best_model.weights_[old]),
                "training_hard_count": int(best_counts[old]),
                "weight_expected_rows": float(best_model.weights_[old] * len(X_scaled)),
                "geometric_center_credit_term": float(np.exp(center[0])),
            }
            if len(center) == 2:
                row["geometric_center_credit_to_goods"] = float(np.exp(center[1]))
            components.append(row)

        reference_assignments = best_model.predict(X_scaled)
        stability_rows = []
        aris = []
        for repeat in range(1, GMM_STABILITY_REPEATS + 1):
            restart = GaussianMixture(
                n_components=best_k,
                covariance_type="full",
                reg_covar=GMM_REG_COVAR,
                max_iter=GMM_MAX_ITER,
                n_init=1,
                init_params="kmeans",
                random_state=(
                    GMM_RANDOM_STATE
                    + 900000
                    + 1000 * self.fold_number
                    + 100 * best_k
                    + repeat
                ),
            )
            restart.fit(X_scaled)
            restart_labels = restart.predict(X_scaled)
            restart_counts = np.bincount(restart_labels, minlength=best_k)
            ari = float(adjusted_rand_score(reference_assignments, restart_labels))
            aris.append(ari)
            stability_rows.append(
                {
                    "repeat": int(repeat),
                    "converged": bool(restart.converged_),
                    "iterations": int(restart.n_iter_),
                    "adjusted_rand_vs_selected": ari,
                    "minimum_hard_cluster_rows": int(restart_counts.min()),
                    "maximum_hard_cluster_rows": int(restart_counts.max()),
                    "all_components_meet_minimum_support": bool(
                        restart_counts.min() >= GMM_MIN_COMPONENT_ROWS
                    ),
                }
            )

        metadata = {
            "branch": branch_name,
            "input_dimensions": int(X_raw.shape[1]),
            "training_rows": int(len(X_raw)),
            "selected_n_components": int(best_k),
            "selected_bic": float(best_bic),
            "selected_at_search_boundary": bool(best_k == max_k),
            "covariance_type": "full",
            "gmm_min_component_rows": GMM_MIN_COMPONENT_ROWS,
            "max_components_searched": int(max_k),
            "standard_scaler_mean": scaler.mean_.tolist(),
            "standard_scaler_scale": scaler.scale_.tolist(),
            "component_details_sorted_by_center": components,
            "bic_search": candidate_rows,
            "candidate_component_sizes": candidate_sizes,
            "initialization_stability": {
                "repeats": GMM_STABILITY_REPEATS,
                "mean_ari_vs_selected": float(np.mean(aris)),
                "minimum_ari_vs_selected": float(np.min(aris)),
                "maximum_ari_vs_selected": float(np.max(aris)),
                "restart_details": stability_rows,
            },
        }
        return scaler, best_model, rank_map, metadata

    def fit(self, frame: pd.DataFrame):
        term, goods = self.calculate_ratios(frame)
        t = term.to_numpy(dtype=np.float64)
        g = goods.to_numpy(dtype=np.float64)
        term_valid = np.isfinite(t) & (t > 0)
        goods_valid = np.isfinite(g) & (g > 0)

        self.term_spike_info = self.detect_spike(
            t, LOAN_STRUCTURE_TERM_PRECISION, "CREDIT_TERM"
        )
        self.goods_spike_info = self.detect_spike(
            g, LOAN_STRUCTURE_GOODS_PRECISION, "CREDIT_TO_GOODS"
        )
        term_spike = self.spike_mask(t, self.term_spike_info, LOAN_STRUCTURE_TERM_PRECISION)
        goods_spike = self.spike_mask(g, self.goods_spike_info, LOAN_STRUCTURE_GOODS_PRECISION)

        both_valid = term_valid & goods_valid
        both = both_valid & term_spike & goods_spike
        term_only = both_valid & term_spike & ~goods_spike
        goods_only = both_valid & ~term_spike & goods_spike
        neither = both_valid & ~term_spike & ~goods_spike

        self.training_branch_counts = {
            "BOTH_SPIKES": int(both.sum()),
            "TERM_SPIKE_ONLY": int(term_only.sum()),
            "GOODS_SPIKE_ONLY": int(goods_only.sum()),
            "NEITHER_SPIKE_2D": int(neither.sum()),
            "MISSING_TERM": int((~term_valid & goods_valid).sum()),
            "MISSING_GOODS": int((term_valid & ~goods_valid).sum()),
            "MISSING_BOTH": int((~term_valid & ~goods_valid).sum()),
        }

        if goods_only.sum() < GMM_MIN_COMPONENT_ROWS:
            raise RuntimeError(f"GOODS_SPIKE_ONLY has only {int(goods_only.sum())} rows")
        if neither.sum() < GMM_MIN_COMPONENT_ROWS:
            raise RuntimeError(f"NEITHER_SPIKE_2D has only {int(neither.sum())} rows")

        (
            self.term_gmm_scaler,
            self.term_gmm_model,
            self.term_gmm_rank_map,
            self.term_gmm_metadata,
        ) = self.fit_gmm_by_bic(
            np.log(t[goods_only]).reshape(-1, 1),
            "GOODS_SPIKE_ONLY__1D_GMM_ON_LOG_CREDIT_TERM",
        )
        (
            self.joint_gmm_scaler,
            self.joint_gmm_model,
            self.joint_gmm_rank_map,
            self.joint_gmm_metadata,
        ) = self.fit_gmm_by_bic(
            np.column_stack([np.log(t[neither]), np.log(g[neither])]),
            "NEITHER_SPIKE__2D_GMM_ON_BOTH_LOG_RATIOS",
        )
        self.training_support = {
            **self.training_branch_counts,
            "TERM_SPIKE_ONLY_FURTHER_SPLIT_REVIEW_THRESHOLD": TERM_SPIKE_ONLY_REVIEW_SUPPORT,
            "TERM_SPIKE_ONLY_HAS_SUPPORT_FOR_REVIEW": bool(
                term_only.sum() >= TERM_SPIKE_ONLY_REVIEW_SUPPORT
            ),
        }
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise RuntimeError("Segmenter must be fitted before transform")
        term, goods = self.calculate_ratios(frame)
        t = term.to_numpy(dtype=np.float64)
        g = goods.to_numpy(dtype=np.float64)
        term_valid = np.isfinite(t) & (t > 0)
        goods_valid = np.isfinite(g) & (g > 0)
        term_spike = self.spike_mask(t, self.term_spike_info, LOAN_STRUCTURE_TERM_PRECISION)
        goods_spike = self.spike_mask(g, self.goods_spike_info, LOAN_STRUCTURE_GOODS_PRECISION)

        both_valid = term_valid & goods_valid
        both = both_valid & term_spike & goods_spike
        term_only = both_valid & term_spike & ~goods_spike
        goods_only = both_valid & ~term_spike & goods_spike
        neither = both_valid & ~term_spike & ~goods_spike

        labels = np.full(len(frame), "MISSING_BOTH", dtype=object)
        labels[~term_valid & goods_valid] = "MISSING_TERM"
        labels[term_valid & ~goods_valid] = "MISSING_GOODS"
        labels[both] = "BOTH_SPIKES"
        labels[term_only] = "TERM_SPIKE_ONLY"

        if goods_only.any():
            scaled = self.term_gmm_scaler.transform(np.log(t[goods_only]).reshape(-1, 1))
            ranks = self.term_gmm_rank_map[self.term_gmm_model.predict(scaled)]
            labels[goods_only] = [f"GOODS_SPIKE_TERM_GMM_{rank:02d}" for rank in ranks]
        if neither.any():
            scaled = self.joint_gmm_scaler.transform(
                np.column_stack([np.log(t[neither]), np.log(g[neither])])
            )
            ranks = self.joint_gmm_rank_map[self.joint_gmm_model.predict(scaled)]
            labels[neither] = [f"GMM2D_{rank:02d}" for rank in ranks]
        return pd.Series(labels, index=frame.index, name=LOAN_STRUCTURE_FEATURE, dtype="object")

    def metadata(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("Cannot serialize an unfitted segmenter")
        return {
            "fold": self.fold_number,
            "feature_name": LOAN_STRUCTURE_FEATURE,
            "segmentation_method": "Hybrid spike handling + BIC-selected GaussianMixture",
            "target_used": False,
            "spike_detection": {
                "minimum_share": LOAN_STRUCTURE_SPIKE_MIN_SHARE,
                "top_to_second_frequency_ratio": LOAN_STRUCTURE_SPIKE_DOMINANCE,
                "term_precision": LOAN_STRUCTURE_TERM_PRECISION,
                "goods_precision": LOAN_STRUCTURE_GOODS_PRECISION,
                "CREDIT_TERM": self.term_spike_info,
                "CREDIT_TO_GOODS": self.goods_spike_info,
            },
            "branch_training_counts": self.training_branch_counts,
            "term_spike_only_support_review": self.training_support,
            "goods_spike_only_1d_gmm": self.term_gmm_metadata,
            "neither_spike_2d_gmm": self.joint_gmm_metadata,
            "gmm_selection": (
                "minimum BIC among converged candidates meeting minimum hard-cluster support"
            ),
        }


# =============================================================================
# 7.1. PERSISTENT GMM CACHE — SAME PATH / FINGERPRINT STRATEGY AS OLD SCRIPT
# =============================================================================


def build_gmm_cache_key(
    fold_number: int,
    fit_indices: np.ndarray,
    X_train_raw: pd.DataFrame,
    train_id_values: np.ndarray,
) -> str:
    """Fingerprint exact fold membership, GMM predictors, settings, and versions."""
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    train_id_values = np.asarray(train_id_values)
    missing = [column for column in GMM_CACHE_INPUT_COLUMNS if column not in X_train_raw.columns]
    if missing:
        raise KeyError(f"GMM cache fingerprint missing columns: {missing}")

    source = X_train_raw.iloc[fit_indices][GMM_CACHE_INPUT_COLUMNS].copy()
    identity = pd.DataFrame({ID_COL: train_id_values[fit_indices]})
    for column in GMM_CACHE_INPUT_COLUMNS:
        identity[column] = source[column].to_numpy()

    configuration = {
        "cache_schema_version": GMM_CACHE_SCHEMA_VERSION,
        "fold_number": int(fold_number),
        "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "input_columns": GMM_CACHE_INPUT_COLUMNS,
        "spike_detection": {
            "term_precision": LOAN_STRUCTURE_TERM_PRECISION,
            "goods_precision": LOAN_STRUCTURE_GOODS_PRECISION,
            "minimum_share": LOAN_STRUCTURE_SPIKE_MIN_SHARE,
            "spike_dominance": LOAN_STRUCTURE_SPIKE_DOMINANCE,
            "term_spike_only_review_support": TERM_SPIKE_ONLY_REVIEW_SUPPORT,
        },
        "gmm": {
            "max_components": GMM_MAX_COMPONENTS,
            "n_init": GMM_N_INIT,
            "max_iter": GMM_MAX_ITER,
            "reg_covar": GMM_REG_COVAR,
            "minimum_component_rows": GMM_MIN_COMPONENT_ROWS,
            "random_state": GMM_RANDOM_STATE,
            "stability_repeats": GMM_STABILITY_REPEATS,
            "covariance_type": "full",
            "init_params": "kmeans",
        },
        "library_versions": {
            "python": sys.version.split()[0],
            "scikit_learn": get_package_version("scikit-learn"),
            "scipy": get_package_version("scipy"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    digest = hashlib.sha256()
    config_bytes = json.dumps(
        configuration,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest.update(config_bytes)
    digest.update(fit_indices.astype("<i8", copy=False).tobytes())
    row_hashes = pd.util.hash_pandas_object(
        identity, index=False, categorize=True
    ).to_numpy(dtype=np.uint64)
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def atomic_pickle_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary_path.open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def load_cached_gmm_segmenter(
    cache_path: Path,
    cache_key: str,
    fold_number: int,
) -> HybridLoanStructureSegmenter | None:
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as file:
            payload = pickle.load(file)
    except Exception as exc:
        print(f"[GMM CACHE WARNING] Could not load {cache_path.name}: {exc}", flush=True)
        return None
    if not isinstance(payload, dict):
        print(f"[GMM CACHE WARNING] Invalid cache structure: {cache_path.name}", flush=True)
        return None
    if payload.get("cache_schema_version") != GMM_CACHE_SCHEMA_VERSION:
        print(f"[GMM CACHE] Schema mismatch for fold {fold_number}; refitting.", flush=True)
        return None
    if payload.get("cache_key") != cache_key:
        print(f"[GMM CACHE] Fingerprint mismatch for fold {fold_number}; refitting.", flush=True)
        return None
    if payload.get("fold_number") != fold_number:
        print(f"[GMM CACHE] Fold mismatch for fold {fold_number}; refitting.", flush=True)
        return None
    segmenter = payload.get("segmenter")
    if not isinstance(segmenter, HybridLoanStructureSegmenter):
        print(f"[GMM CACHE] Invalid segmenter object for fold {fold_number}; refitting.", flush=True)
        return None
    if not getattr(segmenter, "fitted", False) or segmenter.fold_number != fold_number:
        print(f"[GMM CACHE] Unfitted / wrong-fold object for fold {fold_number}; refitting.", flush=True)
        return None
    return segmenter


def save_cached_gmm_segmenter(
    segmenter: HybridLoanStructureSegmenter,
    cache_path: Path,
    cache_key: str,
) -> None:
    if not segmenter.fitted:
        raise RuntimeError("Cannot save an unfitted GMM segmenter")
    payload = {
        "cache_schema_version": GMM_CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "fold_number": segmenter.fold_number,
        "segmenter": segmenter,
    }
    atomic_pickle_dump(payload, cache_path)
    print(f"[GMM CACHE SAVED] Fold {segmenter.fold_number}: {cache_path}", flush=True)


def get_fold_gmm_segmenter(
    fold_number: int,
    fit_indices: np.ndarray,
    X_train_raw: pd.DataFrame,
    train_id_values: np.ndarray,
) -> tuple[HybridLoanStructureSegmenter, dict[str, Any], str]:
    """Restore run checkpoint, else persistent cache, else fit and cache once."""
    run_fold_dir = GMM_DIR / f"FOLD_{fold_number:02d}"
    run_fold_dir.mkdir(parents=True, exist_ok=True)
    run_model_path = run_fold_dir / "segmenter.pkl"
    metadata_path = run_fold_dir / "segmenter_metadata.json"
    run_marker = run_fold_dir / "GMM_COMPLETE.json"

    cache_key = build_gmm_cache_key(
        fold_number, fit_indices, X_train_raw, train_id_values
    )
    persistent_cache_path = GMM_CACHE_DIR / (
        f"hybrid_gmm_fold_{fold_number:02d}_{cache_key}.pkl"
    )

    if checkpoint_is_complete(run_marker, "gmm_segmenter", CONFIG_HASH):
        if not run_model_path.exists() or not metadata_path.exists():
            raise RuntimeError(f"Fold {fold_number}: GMM run checkpoint is incomplete")
        segmenter = load_pickle(run_model_path)
        if (
            not isinstance(segmenter, HybridLoanStructureSegmenter)
            or not getattr(segmenter, "fitted", False)
            or segmenter.fold_number != fold_number
        ):
            raise RuntimeError(f"Fold {fold_number}: saved run GMM segmenter is invalid")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(f"Fold {fold_number}: restored run-specific GMM checkpoint.", flush=True)
        return segmenter, metadata, "RUN_CHECKPOINT"

    segmenter = load_cached_gmm_segmenter(
        persistent_cache_path, cache_key, fold_number
    )
    if segmenter is not None:
        cache_status = "LOADED_FROM_PERSISTENT_CACHE"
        print(f"[GMM CACHE HIT] Fold {fold_number}: loaded persistent segmenter.", flush=True)
    else:
        cache_status = "FITTED_AND_SAVED"
        print(f"[GMM CACHE MISS] Fold {fold_number}: fitting GMM segmenter.", flush=True)
        segmenter = HybridLoanStructureSegmenter(fold_number)
        with threadpool_limits(limits=CPU_THREADS):
            segmenter.fit(X_train_raw.iloc[fit_indices])
        save_cached_gmm_segmenter(segmenter, persistent_cache_path, cache_key)

    metadata = segmenter.metadata()
    atomic_pickle_dump(segmenter, run_model_path)
    atomic_json(metadata_path, metadata)
    mark_complete(
        run_marker,
        "gmm_segmenter",
        CONFIG_HASH,
        [run_model_path, metadata_path],
    )
    status_path = run_fold_dir / "GMM_CACHE_STATUS.json"
    atomic_json(
        status_path,
        {
            "fold": fold_number,
            "status": cache_status,
            "cache_key": cache_key,
            "persistent_cache_path": str(persistent_cache_path),
            "run_model_path": str(run_model_path),
            "training_rows": int(len(fit_indices)),
            "validation_rows": int(n_train - len(fit_indices)),
            "selected_1d_k": int(segmenter.term_gmm_metadata["selected_n_components"]),
            "selected_2d_k": int(segmenter.joint_gmm_metadata["selected_n_components"]),
        },
    )
    return segmenter, metadata, cache_status


# =============================================================================
# 8. LOCAL 500-NEIGHBOUR INTERACTION — FROM IMPUTED EXT_SOURCE VALUES
# =============================================================================


def extract_ext_source_sum_and_age(
    frame: pd.DataFrame,
    age_imputation_value: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    source_arrays = [
        pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        for column in EXT_SOURCE_COLUMNS
    ]
    source_matrix = np.column_stack(source_arrays)
    if not np.isfinite(source_matrix).all():
        raise ValueError("EXT_SOURCE columns must be fully imputed before local interaction")
    ext_sum = source_matrix.sum(axis=1)

    days_birth = pd.to_numeric(frame["DAYS_BIRTH"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    age = (-days_birth / 365.25).to_numpy(dtype=np.float64)
    if age_imputation_value is None:
        finite_age = age[np.isfinite(age)]
        if finite_age.size == 0:
            raise ValueError("DAYS_BIRTH has no finite values in fold-training data")
        age_imputation_value = float(np.median(finite_age))
    age = np.where(np.isfinite(age), age, float(age_imputation_value))
    return ext_sum, age, float(age_imputation_value)


def compute_training_neighbor_means(
    reference_coords: np.ndarray,
    reference_ext: np.ndarray,
    reference_age: np.ndarray,
    neighbor_model: NearestNeighbors,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(reference_ext)
    if n <= N_NEIGHBORS:
        raise ValueError(f"Need more than {N_NEIGHBORS} fold-training rows; received {n}")
    mean_ext = np.empty(n, dtype=np.float64)
    mean_age = np.empty(n, dtype=np.float64)
    n_query = min(N_NEIGHBORS + 2, n)

    for start in range(0, n, NEIGHBOR_QUERY_CHUNK_SIZE):
        stop = min(start + NEIGHBOR_QUERY_CHUNK_SIZE, n)
        neighbors = neighbor_model.kneighbors(
            reference_coords[start:stop],
            n_neighbors=n_query,
            return_distance=False,
        )
        self_indices = np.arange(start, stop, dtype=np.int64)
        self_match = neighbors == self_indices[:, None]
        has_self = self_match.any(axis=1)
        self_position = np.where(has_self, self_match.argmax(axis=1), n_query)
        output_columns = np.arange(N_NEIGHBORS, dtype=np.int64)[None, :]
        source_columns = output_columns + (output_columns >= self_position[:, None])
        selected = np.take_along_axis(neighbors, source_columns, axis=1)
        if selected.shape[1] != N_NEIGHBORS:
            raise RuntimeError("Training neighbor selection did not return exactly 500 rows")
        mean_ext[start:stop] = reference_ext[selected].mean(axis=1)
        mean_age[start:stop] = reference_age[selected].mean(axis=1)
    return mean_ext, mean_age


def compute_query_neighbor_means(
    query_coords: np.ndarray,
    reference_ext: np.ndarray,
    reference_age: np.ndarray,
    neighbor_model: NearestNeighbors,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(query_coords)
    mean_ext = np.empty(n, dtype=np.float64)
    mean_age = np.empty(n, dtype=np.float64)
    for start in range(0, n, NEIGHBOR_QUERY_CHUNK_SIZE):
        stop = min(start + NEIGHBOR_QUERY_CHUNK_SIZE, n)
        neighbors = neighbor_model.kneighbors(
            query_coords[start:stop],
            n_neighbors=N_NEIGHBORS,
            return_distance=False,
        )
        mean_ext[start:stop] = reference_ext[neighbors].mean(axis=1)
        mean_age[start:stop] = reference_age[neighbors].mean(axis=1)
    return mean_ext, mean_age


def build_fold_local_neighbor_interaction(
    X_fit_raw: pd.DataFrame,
    X_valid_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, pd.DataFrame]]:
    fit_ext, fit_age, age_median = extract_ext_source_sum_and_age(X_fit_raw)
    valid_ext, valid_age, _ = extract_ext_source_sum_and_age(X_valid_raw, age_median)
    test_ext, test_age, _ = extract_ext_source_sum_and_age(X_test_raw, age_median)

    fit_2d = np.column_stack([fit_ext, fit_age]).astype(np.float64)
    center = fit_2d.mean(axis=0)
    scale = fit_2d.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    fit_coords = (fit_2d - center) / scale
    valid_coords = (np.column_stack([valid_ext, valid_age]) - center) / scale
    test_coords = (np.column_stack([test_ext, test_age]) - center) / scale

    neighbor_model = NearestNeighbors(
        algorithm="kd_tree",
        metric="euclidean",
        leaf_size=40,
        n_jobs=CPU_THREADS,
    )
    neighbor_model.fit(fit_coords)
    fit_mean_ext, fit_mean_age = compute_training_neighbor_means(
        fit_coords, fit_ext, fit_age, neighbor_model
    )
    valid_mean_ext, valid_mean_age = compute_query_neighbor_means(
        valid_coords, fit_ext, fit_age, neighbor_model
    )
    test_mean_ext, test_mean_age = compute_query_neighbor_means(
        test_coords, fit_ext, fit_age, neighbor_model
    )

    fit_interaction = (fit_ext - fit_mean_ext) * (fit_age - fit_mean_age)
    valid_interaction = (valid_ext - valid_mean_ext) * (valid_age - valid_mean_age)
    test_interaction = (test_ext - test_mean_ext) * (test_age - test_mean_age)
    for split_name, values in [
        ("train", fit_interaction),
        ("validation", valid_interaction),
        ("test", test_interaction),
    ]:
        if not np.isfinite(values).all():
            raise RuntimeError(f"Local interaction contains non-finite values in {split_name}")

    diagnostics = {
        "validation": pd.DataFrame(
            {
                "EXT_SOURCE_SUM_IMPUTED": valid_ext,
                "AGE_YEARS": valid_age,
                "MEAN_EXT_SOURCE_SUM_500_NEIGHBORS": valid_mean_ext,
                "MEAN_AGE_YEARS_500_NEIGHBORS": valid_mean_age,
                NEW_INTERACTION_FEATURE: valid_interaction,
            }
        ),
        "test": pd.DataFrame(
            {
                "EXT_SOURCE_SUM_IMPUTED": test_ext,
                "AGE_YEARS": test_age,
                "MEAN_EXT_SOURCE_SUM_500_NEIGHBORS": test_mean_ext,
                "MEAN_AGE_YEARS_500_NEIGHBORS": test_mean_age,
                NEW_INTERACTION_FEATURE: test_interaction,
            }
        ),
    }
    del neighbor_model, fit_coords, valid_coords, test_coords, fit_2d
    gc.collect()
    return fit_interaction, valid_interaction, test_interaction, diagnostics


# =============================================================================
# 9. TREE + REALMLP FEATURE ENGINEERING
# =============================================================================


def _career_profile_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize only the three raw categorical inputs used by CAREER_PROFILE."""
    required = ["NAME_INCOME_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"CAREER_PROFILE inputs are missing: {missing}")
    return pd.DataFrame(
        {
            "INCOME": frame["NAME_INCOME_TYPE"].astype("string").fillna(MISSING_CATEGORY).astype(str),
            "OCCUPATION": frame["OCCUPATION_TYPE"].astype("string").fillna(MISSING_CATEGORY).astype(str),
            "ORGANIZATION": frame["ORGANIZATION_TYPE"].astype("string").fillna(MISSING_CATEGORY).astype(str),
        },
        index=frame.index,
    )


def _career_key_counts(
    reference_keys: pd.DataFrame,
    query_keys: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    """Count combinations in reference rows and look them up for query rows."""
    count_map = reference_keys.groupby(columns, dropna=False, sort=False).size().to_dict()
    query_rows = query_keys[columns].itertuples(index=False, name=None)
    return np.fromiter(
        (int(count_map.get(tuple(row), 0)) for row in query_rows),
        dtype=np.int64,
        count=len(query_keys),
    )


def build_career_profile(
    reference_frame: pd.DataFrame,
    query_frame: pd.DataFrame,
    min_count: int = CAREER_PROFILE_MIN_COUNT,
) -> pd.Series:
    """Fit hierarchical category support on reference rows, then transform query rows.

    For non-pensioners, use the first sufficiently supported key in this order:
      1) income + occupation + organization;
      2) income + occupation;
      3) income + organization;
      4) income only.
    Pensioners receive a dedicated profile. Missing occupation/organization values
    remain explicit key values; literal XNA organization values are not treated as NA.
    """
    if int(min_count) < 1:
        raise ValueError("CAREER_PROFILE min_count must be >= 1")
    reference_keys = _career_profile_keys(reference_frame)
    query_keys = _career_profile_keys(query_frame)

    triple_count = _career_key_counts(
        reference_keys, query_keys, ["INCOME", "OCCUPATION", "ORGANIZATION"]
    )
    income_occupation_count = _career_key_counts(
        reference_keys, query_keys, ["INCOME", "OCCUPATION"]
    )
    income_organization_count = _career_key_counts(
        reference_keys, query_keys, ["INCOME", "ORGANIZATION"]
    )

    income = query_keys["INCOME"].to_numpy(dtype=object)
    occupation = query_keys["OCCUPATION"].to_numpy(dtype=object)
    organization = query_keys["ORGANIZATION"].to_numpy(dtype=object)
    pensioner = income == "Pensioner"

    # Lowest-resolution fallback is always defined, including unseen income values.
    values = np.asarray([f"INCOME::{value}" for value in income], dtype=object)

    use_income_occupation_organization = (~pensioner) & (triple_count >= int(min_count))
    use_income_occupation = (
        (~pensioner)
        & (~use_income_occupation_organization)
        & (income_occupation_count >= int(min_count))
    )
    use_income_organization = (
        (~pensioner)
        & (~use_income_occupation_organization)
        & (~use_income_occupation)
        & (income_organization_count >= int(min_count))
    )

    for i in np.flatnonzero(use_income_occupation_organization):
        values[i] = f"INCOME_OCCUPATION_ORGANIZATION::{income[i]}::{occupation[i]}::{organization[i]}"
    for i in np.flatnonzero(use_income_occupation):
        values[i] = f"INCOME_OCCUPATION::{income[i]}::{occupation[i]}"
    for i in np.flatnonzero(use_income_organization):
        values[i] = f"INCOME_ORGANIZATION::{income[i]}::{organization[i]}"
    values[pensioner] = CAREER_PROFILE_PENSIONER_VALUE

    return pd.Series(values, index=query_frame.index, name=CAREER_PROFILE_FEATURE, dtype="string")


def build_inner_career_profiles(
    X_raw: pd.DataFrame,
    train_positions: np.ndarray,
    valid_positions: np.ndarray,
) -> tuple[pd.Series, pd.Series]:
    """Re-fit CAREER_PROFILE support on an inner-training split only."""
    reference = X_raw.iloc[train_positions]
    query_train = X_raw.iloc[train_positions]
    query_valid = X_raw.iloc[valid_positions]
    return (
        build_career_profile(reference, query_train),
        build_career_profile(reference, query_valid),
    )


def replace_mlp_career_profile_codes(
    X_train_encoded: pd.DataFrame,
    X_query_encoded: pd.DataFrame,
    train_profiles: pd.Series,
    query_profiles: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace the profile codes using a vocabulary fitted on this split's train rows."""
    if CAREER_PROFILE_FEATURE not in X_train_encoded.columns or CAREER_PROFILE_FEATURE not in X_query_encoded.columns:
        raise KeyError(f"{CAREER_PROFILE_FEATURE} is absent from the RealMLP feature view")
    train_out = X_train_encoded.copy()
    query_out = X_query_encoded.copy()
    train_values = pd.Series(
        train_profiles.astype("string").to_numpy(), index=train_out.index, dtype="string"
    )
    query_values = pd.Series(
        query_profiles.astype("string").to_numpy(), index=query_out.index, dtype="string"
    )
    vocabulary = sorted(train_values.dropna().unique().tolist())
    mapping = {value: i + 1 for i, value in enumerate(vocabulary)}
    unknown_code = len(mapping) + 1
    train_codes = train_values.map(mapping).fillna(0).astype(np.int32)
    query_codes = query_values.map(mapping).fillna(unknown_code).astype(np.int32)
    query_codes.loc[query_values.isna()] = 0
    train_out[CAREER_PROFILE_FEATURE] = train_codes
    query_out[CAREER_PROFILE_FEATURE] = query_codes
    return train_out, query_out


def fit_tree_feature_statistics(X_fit: pd.DataFrame) -> dict[str, pd.Series]:
    income = pd.to_numeric(X_fit["AMT_INCOME_TOTAL"], errors="coerce")
    region = pd.to_numeric(X_fit["REGION_RATING_CLIENT"], errors="coerce")
    regional_means = (
        pd.DataFrame({"REGION": region, "INCOME": income})
        .dropna(subset=["REGION"])
        .groupby("REGION")["INCOME"]
        .mean()
    )
    return {"regional_means": regional_means}


def add_tree_engineered_features(
    frame: pd.DataFrame,
    statistics: dict[str, Any],
) -> pd.DataFrame:
    output = frame.copy()
    
    credit = pd.to_numeric(output["AMT_CREDIT"], errors="coerce")
    annuity_raw = pd.to_numeric(output["AMT_ANNUITY"], errors="coerce")
    annuity = annuity_raw.replace(0, np.nan)
    goods = pd.to_numeric(output["AMT_GOODS_PRICE"], errors="coerce").replace(0, np.nan)
    income = pd.to_numeric(output["AMT_INCOME_TOTAL"], errors="coerce")
    family_members = pd.to_numeric(output["CNT_FAM_MEMBERS"], errors="coerce")
    region = pd.to_numeric(output["REGION_RATING_CLIENT"], errors="coerce")
    car_age = pd.to_numeric(output["OWN_CAR_AGE"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )

    output["CREDIT_TERM"] = safe_ratio(credit, annuity)
    regional_mean = region.map(statistics["regional_means"]).replace(0, np.nan)
    output["GOODS_PRICE_TO_ANNUITY"] = safe_ratio(goods, annuity)
    output["INCOME_TO_REGION_MEAN"] = safe_ratio(income, regional_mean)
    output["OWN_CAR_AGE_CLEAN"] = car_age.fillna(-1.0)
    output["DIFF_RATIO"] = safe_ratio(credit - goods, goods)
    output["NET_INCOME_PER_MEMBER"] = safe_ratio(income - annuity_raw, family_members)
    output["CAR_FRESHNESS"] = (1.0 / (1.0 + car_age.clip(lower=0.0))).replace(
        [np.inf, -np.inf], np.nan
    )
    numeric_added = [
        "CREDIT_TERM",
        "INCOME_TO_REGION_MEAN",
        "OWN_CAR_AGE_CLEAN",
        "DIFF_RATIO",
        "NET_INCOME_PER_MEMBER",
        "CAR_FRESHNESS",
    ]
    output[numeric_added] = output[numeric_added].replace([np.inf, -np.inf], np.nan)
    return output


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype(np.float64)


def fit_mlp_region_income_means(
    reference_df: pd.DataFrame,
) -> tuple[pd.Series, float]:
    income = numeric_column(reference_df, "AMT_INCOME_TOTAL")
    region = reference_df["REGION_RATING_CLIENT"].astype("string").fillna("__MISSING_REGION__")
    work = pd.DataFrame({"region": region.to_numpy(), "income": income.to_numpy()})
    means = work.groupby("region", dropna=False)["income"].mean()
    global_mean = float(income.mean())
    if not np.isfinite(global_mean) or global_mean == 0:
        global_mean = 1.0
    return means, global_mean


def build_mlp_feature_frame(
    query_df: pd.DataFrame,
    *,
    region_means: pd.Series,
    global_income_mean: float,
    local_interaction: np.ndarray,
    loan_cluster: pd.Series,
    raw_feature_columns: list[str],
) -> pd.DataFrame:
    output = query_df[raw_feature_columns].copy()
    credit = numeric_column(query_df, "AMT_CREDIT")
    annuity = numeric_column(query_df, "AMT_ANNUITY")
    income = numeric_column(query_df, "AMT_INCOME_TOTAL")
    goods_price = numeric_column(query_df, "AMT_GOODS_PRICE")
    family_members = numeric_column(query_df, "CNT_FAM_MEMBERS")
    car_age = numeric_column(query_df, "OWN_CAR_AGE").replace([np.inf, -np.inf], np.nan)
    output["GOODS_PRICE_TO_ANNUITY"] = safe_ratio(goods_price, annuity)

    output["CREDIT_TERM"] = safe_ratio(credit, annuity)
    region_key = query_df["REGION_RATING_CLIENT"].astype("string").fillna("__MISSING_REGION__")
    region_mean_for_row = region_key.map(region_means).astype(np.float64).fillna(global_income_mean)
    output["INCOME_TO_REGION_MEAN"] = safe_ratio(income, region_mean_for_row)
    output["OWN_CAR_AGE_CLEAN"] = car_age.fillna(-1.0)
    output["DIFF_RATIO"] = safe_ratio(credit - goods_price, goods_price)
    output["NET_INCOME_PER_MEMBER"] = safe_ratio(income - annuity, family_members)
    output["CAR_FRESHNESS"] = (1.0 / (1.0 + car_age.clip(lower=0.0))).replace(
        [np.inf, -np.inf], np.nan
    )

    if len(local_interaction) != len(output) or len(loan_cluster) != len(output):
        raise ValueError("RealMLP engineered arrays have incorrect row counts")
    output[NEW_INTERACTION_FEATURE] = local_interaction
    output[LOAN_STRUCTURE_FEATURE] = loan_cluster.astype("string")

    numeric_columns = [column for column in output.columns if column not in MLP_CATEGORICAL_COLUMNS]
    output[numeric_columns] = output[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return output


def encode_mlp_fold_inputs(
    X_fit_raw: pd.DataFrame,
    X_valid_raw: pd.DataFrame,
    X_test_raw_fold: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], dict[str, Any]]:
    """Fit numeric medians and category vocabularies on this fold's training rows only."""
    for column in MLP_CATEGORICAL_COLUMNS:
        if column not in X_fit_raw.columns:
            raise KeyError(f"MLP categorical column missing: {column}")

    numeric_columns = [
        column for column in X_fit_raw.columns if column not in MLP_CATEGORICAL_COLUMNS
    ]
    fit_num = X_fit_raw[numeric_columns].copy()
    valid_num = X_valid_raw[numeric_columns].copy()
    test_num = X_test_raw_fold[numeric_columns].copy()
    for frame in (fit_num, valid_num, test_num):
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)

    medians = fit_num.median(axis=0, skipna=True).fillna(0.0)
    missing_indicator_columns = [
        column for column in numeric_columns if fit_num[column].isna().any()
    ]
    fit_missing = fit_num[missing_indicator_columns].isna().astype(np.float32)
    valid_missing = valid_num[missing_indicator_columns].isna().astype(np.float32)
    test_missing = test_num[missing_indicator_columns].isna().astype(np.float32)

    fit_num = fit_num.fillna(medians).astype(np.float32)
    valid_num = valid_num.fillna(medians).astype(np.float32)
    test_num = test_num.fillna(medians).astype(np.float32)
    if missing_indicator_columns:
        indicator_names = [f"{column}__IS_MISSING" for column in missing_indicator_columns]
        fit_missing.columns = indicator_names
        valid_missing.columns = indicator_names
        test_missing.columns = indicator_names
        fit_num = pd.concat([fit_num, fit_missing], axis=1)
        valid_num = pd.concat([valid_num, valid_missing], axis=1)
        test_num = pd.concat([test_num, test_missing], axis=1)

    fit_cat = pd.DataFrame(index=X_fit_raw.index)
    valid_cat = pd.DataFrame(index=X_valid_raw.index)
    test_cat = pd.DataFrame(index=X_test_raw_fold.index)
    category_maps: dict[str, Any] = {}
    for column in MLP_CATEGORICAL_COLUMNS:
        fit_values = X_fit_raw[column].astype("string")
        valid_values = X_valid_raw[column].astype("string")
        test_values = X_test_raw_fold[column].astype("string")
        vocabulary = sorted(fit_values.dropna().unique().tolist())
        mapping = {value: index + 1 for index, value in enumerate(vocabulary)}
        unknown_code = len(mapping) + 1
        fit_codes = fit_values.map(mapping).fillna(0).astype(np.int32)
        valid_codes = valid_values.map(mapping).fillna(unknown_code).astype(np.int32)
        test_codes = test_values.map(mapping).fillna(unknown_code).astype(np.int32)
        valid_codes.loc[valid_values.isna()] = 0
        test_codes.loc[test_values.isna()] = 0
        fit_cat[column] = fit_codes
        valid_cat[column] = valid_codes
        test_cat[column] = test_codes
        category_maps[column] = {
            "vocabulary": vocabulary,
            "unknown_code": unknown_code,
            "missing_code": 0,
        }

    fit_out = pd.concat([fit_num, fit_cat], axis=1)
    valid_out = pd.concat([valid_num, valid_cat], axis=1)
    test_out = pd.concat([test_num, test_cat], axis=1)
    valid_out = valid_out[fit_out.columns]
    test_out = test_out[fit_out.columns]
    preprocessing = {
        "numeric_medians": {key: float(value) for key, value in medians.items()},
        "numeric_missing_indicator_columns": missing_indicator_columns,
        "categorical_maps": category_maps,
        "categorical_columns": MLP_CATEGORICAL_COLUMNS,
        "output_columns": list(fit_out.columns),
    }
    return (
        fit_out,
        valid_out,
        test_out,
        numeric_columns,
        missing_indicator_columns,
        preprocessing,
    )


def prepare_catboost_data(
    frame: pd.DataFrame,
    categorical_columns: list[str],
) -> pd.DataFrame:
    result = frame.copy()
    for column in categorical_columns:
        result[column] = result[column].astype("string").fillna(MISSING_CATEGORY).astype(str)
    return result


def prepare_gbdt_categorical_views(
    X_fit: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test_fold: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit category vocabularies on fold-train only; unseen categories map to NaN."""
    fit_view, valid_view, test_view = X_fit.copy(), X_valid.copy(), X_test_fold.copy()
    for column in categorical_columns:
        fit_values = X_fit[column].astype("string").fillna(MISSING_CATEGORY)
        valid_values = X_valid[column].astype("string").fillna(MISSING_CATEGORY)
        test_values = X_test_fold[column].astype("string").fillna(MISSING_CATEGORY)
        categories = pd.Index(pd.unique(fit_values))
        fit_view[column] = pd.Categorical(fit_values, categories=categories)
        valid_view[column] = pd.Categorical(valid_values, categories=categories)
        test_view[column] = pd.Categorical(test_values, categories=categories)
    return fit_view, valid_view, test_view


# =============================================================================
# 10. BUILD / RESTORE PER-FOLD FEATURE CACHE
# =============================================================================


def load_feature_cache(fold_number: int) -> dict[str, Any]:
    cache_path = CACHE_DIR / f"FOLD_{fold_number:02d}" / "FEATURE_CACHE.pkl"
    marker_path = CACHE_DIR / f"FOLD_{fold_number:02d}" / "FEATURE_CACHE_COMPLETE.json"
    if not checkpoint_is_complete(marker_path, "feature_cache", CONFIG_HASH):
        raise RuntimeError(f"Fold {fold_number}: feature cache is not marked complete")
    if not cache_path.exists():
        raise FileNotFoundError(f"Fold {fold_number}: feature cache file missing")
    return load_pickle(cache_path)


def build_one_fold_feature_cache(
    fold_number: int,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
) -> None:
    fold_cache_dir = CACHE_DIR / f"FOLD_{fold_number:02d}"
    fold_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = fold_cache_dir / "FEATURE_CACHE.pkl"
    marker_path = fold_cache_dir / "FEATURE_CACHE_COMPLETE.json"

    if checkpoint_is_complete(marker_path, "feature_cache", CONFIG_HASH):
        if not cache_path.exists():
            raise RuntimeError(f"Fold {fold_number}: feature-cache marker exists but file is missing")
        print(f"Fold {fold_number}: restored completed feature cache.", flush=True)
        return

    print_section(f"FEATURE PIPELINE — FOLD {fold_number}/{N_SPLITS}")
    train_imputed = load_and_validate_training_imputations(fold_number)
    test_imputed = generate_test_imputations(fold_number, fit_indices)

    X_train_fold_all = augment_with_ext_source_scenario(X_train_original, train_imputed)
    X_test_fold = augment_with_ext_source_scenario(X_test_original, test_imputed)
    X_fit_raw = X_train_fold_all.iloc[fit_indices].copy()
    X_valid_raw = X_train_fold_all.iloc[valid_indices].copy()

    # CAREER_PROFILE support counts are fitted on this outer-training fold only.
    career_profile_fit = build_career_profile(X_fit_raw, X_fit_raw)
    career_profile_valid = build_career_profile(X_fit_raw, X_valid_raw)
    career_profile_test = build_career_profile(X_fit_raw, X_test_fold)
    X_fit_raw[CAREER_PROFILE_FEATURE] = career_profile_fit
    X_valid_raw[CAREER_PROFILE_FEATURE] = career_profile_valid
    X_test_fold[CAREER_PROFILE_FEATURE] = career_profile_test

    # One fitted or restored segmenter per fold. Exact same labels feed both views.
    segmenter, gmm_metadata, gmm_status = get_fold_gmm_segmenter(
        fold_number,
        fit_indices,
        X_train_original,
        train_ids,
    )
    loan_cluster_fit = segmenter.transform(X_fit_raw)
    loan_cluster_valid = segmenter.transform(X_valid_raw)
    loan_cluster_test = segmenter.transform(X_test_fold)

    # Calculate the local interaction once, after EXT_SOURCE imputation.
    with threadpool_limits(limits=CPU_THREADS):
        local_fit, local_valid, local_test, neighbor_diagnostics = (
            build_fold_local_neighbor_interaction(X_fit_raw, X_valid_raw, X_test_fold)
        )

    # Tree view: original augmented predictors + its four numerical engineered
    # features + the shared local interaction + GMM cluster category.
    tree_statistics = fit_tree_feature_statistics(X_fit_raw)
    X_tree_fit = add_tree_engineered_features(X_fit_raw, tree_statistics)
    X_tree_valid = add_tree_engineered_features(X_valid_raw, tree_statistics)
    X_tree_test = add_tree_engineered_features(X_test_fold, tree_statistics)
    X_tree_fit[LOAN_STRUCTURE_FEATURE] = loan_cluster_fit.astype("string")
    X_tree_valid[LOAN_STRUCTURE_FEATURE] = loan_cluster_valid.astype("string")
    X_tree_test[LOAN_STRUCTURE_FEATURE] = loan_cluster_test.astype("string")
    X_tree_fit[NEW_INTERACTION_FEATURE] = local_fit.astype(np.float32)
    X_tree_valid[NEW_INTERACTION_FEATURE] = local_valid.astype(np.float32)
    X_tree_test[NEW_INTERACTION_FEATURE] = local_test.astype(np.float32)

    # RealMLP view: its five original numerical engineered features plus only
    # the same cluster category; no additional tree-only features are copied.
    mlp_region_means, mlp_global_income_mean = fit_mlp_region_income_means(X_fit_raw)
    raw_augmented_columns = list(X_fit_raw.columns)
    X_mlp_fit_raw = build_mlp_feature_frame(
        X_fit_raw,
        region_means=mlp_region_means,
        global_income_mean=mlp_global_income_mean,
        local_interaction=local_fit,
        loan_cluster=loan_cluster_fit,
        raw_feature_columns=raw_augmented_columns,
    )
    X_mlp_valid_raw = build_mlp_feature_frame(
        X_valid_raw,
        region_means=mlp_region_means,
        global_income_mean=mlp_global_income_mean,
        local_interaction=local_valid,
        loan_cluster=loan_cluster_valid,
        raw_feature_columns=raw_augmented_columns,
    )
    X_mlp_test_raw = build_mlp_feature_frame(
        X_test_fold,
        region_means=mlp_region_means,
        global_income_mean=mlp_global_income_mean,
        local_interaction=local_test,
        loan_cluster=loan_cluster_test,
        raw_feature_columns=raw_augmented_columns,
    )
    (
        X_mlp_fit,
        X_mlp_valid,
        X_mlp_test,
        mlp_numeric_columns,
        mlp_missing_columns,
        mlp_preprocessing,
    ) = encode_mlp_fold_inputs(X_mlp_fit_raw, X_mlp_valid_raw, X_mlp_test_raw)

    tree_categorical_columns = [
        column
        for column in X_tree_fit.columns
        if (
            pd.api.types.is_object_dtype(X_tree_fit[column].dtype)
            or pd.api.types.is_string_dtype(X_tree_fit[column].dtype)
            or isinstance(X_tree_fit[column].dtype, pd.CategoricalDtype)
        )
    ]
    for expected_category in MLP_CATEGORICAL_COLUMNS:
        if expected_category not in X_mlp_fit.columns:
            raise KeyError(
                f"Fold {fold_number}: expected RealMLP categorical {expected_category} is absent"
            )

    if not (
        list(X_tree_fit.columns)
        == list(X_tree_valid.columns)
        == list(X_tree_test.columns)
    ):
        raise RuntimeError("Tree feature schema mismatch across train/validation/test")
    if not (
        list(X_mlp_fit.columns)
        == list(X_mlp_valid.columns)
        == list(X_mlp_test.columns)
    ):
        raise RuntimeError("RealMLP feature schema mismatch across train/validation/test")

    # Hard guard: GOODS_PRICE_TO_ANNUITY is not in either classifier feature view.
    # for view_name, view in [
    #     ("tree", X_tree_fit),
    #     ("realmlp", X_mlp_fit_raw),
    # ]:
    #     # if "GOODS_PRICE_TO_ANNUITY" in view.columns:
    #     #     raise RuntimeError(f"GOODS_PRICE_TO_ANNUITY leaked into {view_name} classifier view")

    cache_payload = {
        "fold": fold_number,
        "fit_indices": np.asarray(fit_indices, dtype=np.int64),
        "valid_indices": np.asarray(valid_indices, dtype=np.int64),
        "y_fit": y[fit_indices].astype(np.int32),
        "y_valid": y[valid_indices].astype(np.int32),
        "tree_fit": X_tree_fit,
        "tree_valid": X_tree_valid,
        "tree_test": X_tree_test,
        "tree_categorical_columns": tree_categorical_columns,
        "mlp_fit": X_mlp_fit,
        "mlp_valid": X_mlp_valid,
        "mlp_test": X_mlp_test,
        "mlp_categorical_columns": MLP_CATEGORICAL_COLUMNS,
        "mlp_numeric_columns_before_missing_indicators": mlp_numeric_columns,
        "mlp_missing_indicator_columns": mlp_missing_columns,
        "mlp_preprocessing": mlp_preprocessing,
        "tree_feature_names": list(X_tree_fit.columns),
        "mlp_feature_names_before_encoding": list(X_mlp_fit_raw.columns),
        "mlp_model_input_columns": list(X_mlp_fit.columns),
        "ext_source_scenario": "IMPUTE_EXT_SOURCE_1_2_3",
        "goods_price_to_annuity_classifier_feature": False,
        "gmm_cache_status": gmm_status,
    }
    atomic_pickle(cache_path, cache_payload)

    # Per-fold diagnostics are persisted before writing the completion marker.
    validation_diag = neighbor_diagnostics["validation"].copy()
    validation_diag.insert(0, "FOLD", fold_number)
    validation_diag[ID_COL] = train_ids[valid_indices]
    test_diag = neighbor_diagnostics["test"].copy()
    test_diag.insert(0, "FOLD", fold_number)
    test_diag[ID_COL] = test_ids
    cluster_diag = pd.DataFrame(
        {
            ID_COL: train_ids[valid_indices],
            "FOLD": fold_number,
            LOAN_STRUCTURE_FEATURE: loan_cluster_valid.astype(str).to_numpy(),
            TARGET_COL: y[valid_indices],
        }
    )
    validation_diag_path = fold_cache_dir / "VALIDATION_NEIGHBOR_DIAGNOSTICS.csv"
    test_diag_path = fold_cache_dir / "TEST_NEIGHBOR_DIAGNOSTICS.csv"
    cluster_diag_path = fold_cache_dir / "VALIDATION_CLUSTERS.csv"
    schema_path = fold_cache_dir / "FEATURE_SCHEMA.json"
    gmm_status_path = GMM_DIR / f"FOLD_{fold_number:02d}" / "GMM_CACHE_STATUS.json"

    atomic_csv(validation_diag_path, validation_diag)
    atomic_csv(test_diag_path, test_diag)
    atomic_csv(cluster_diag_path, cluster_diag)
    atomic_json(
        schema_path,
        {
            "fold": fold_number,
            "tree_feature_names": list(X_tree_fit.columns),
            "tree_categorical_columns": tree_categorical_columns,
            "tree_engineered_features": TREE_ENGINEERED_FEATURES,
            "realmlp_feature_names_before_encoding": list(X_mlp_fit_raw.columns),
            "realmlp_model_input_columns_after_encoding": list(X_mlp_fit.columns),
            "realmlp_engineered_features": MLP_ENGINEERED_FEATURES,
            "realmlp_categorical_columns": MLP_CATEGORICAL_COLUMNS,
            "additional_numeric_features": [
                "DIFF_RATIO", "NET_INCOME_PER_MEMBER", "CAR_FRESHNESS"
            ],
            "career_profile": {
                "feature": CAREER_PROFILE_FEATURE,
                "minimum_training_count": CAREER_PROFILE_MIN_COUNT,
                "pensioner_override": CAREER_PROFILE_PENSIONER_VALUE,
                "fallback_order": [
                    "INCOME+OCCUPATION+ORGANIZATION",
                    "INCOME+OCCUPATION",
                    "INCOME+ORGANIZATION",
                    "INCOME",
                ],
                "fit_scope": "outer training only; re-fitted in every Optuna inner split and RealMLP epoch-selection split",
                "original_categorical_features_retained": True,
            },
            "shared_features": (
                EXT_SOURCE_COLUMNS
                + EXT_SOURCE_MISSING_COLUMNS
                + REQUESTED_SHARED_FEATURES
                + [
                    "DIFF_RATIO", "NET_INCOME_PER_MEMBER", "CAR_FRESHNESS",
                    NEW_INTERACTION_FEATURE, LOAN_STRUCTURE_FEATURE, CAREER_PROFILE_FEATURE,
                ]
            ),
            "goods_price_to_annuity": {
                "classifier_feature": True,
                "imputer_input_feature": True,
            },
            "local_interaction_uses_imputed_ext_source": True,
            "gmm_fit_once_per_fold_and_shared": True,
            "gmm_cache_status": gmm_status,
            "gmm_metadata": gmm_metadata,
        },
    )
    artifacts = [
        cache_path,
        validation_diag_path,
        test_diag_path,
        cluster_diag_path,
        schema_path,
    ]
    if gmm_status_path.exists():
        artifacts.append(gmm_status_path)
    mark_complete(marker_path, "feature_cache", CONFIG_HASH, artifacts)

    print(
        f"Fold {fold_number}: cache saved | Tree={X_tree_fit.shape[1]} columns "
        f"| RealMLP={X_mlp_fit.shape[1]} encoded columns | GMM={gmm_status}",
        flush=True,
    )

    del (
        train_imputed,
        test_imputed,
        X_train_fold_all,
        X_test_fold,
        X_fit_raw,
        X_valid_raw,
        segmenter,
        X_tree_fit,
        X_tree_valid,
        X_tree_test,
        X_mlp_fit_raw,
        X_mlp_valid_raw,
        X_mlp_test_raw,
        X_mlp_fit,
        X_mlp_valid,
        X_mlp_test,
        cache_payload,
        neighbor_diagnostics,
        validation_diag,
        test_diag,
        loan_cluster_fit,
        loan_cluster_valid,
        loan_cluster_test,
    )
    gc.collect()


print_section("BUILDING / RESTORING FOLD-LOCAL IMPUTATION, GMM, AND FEATURE CACHES")
for fold_number, (fit_indices, valid_indices) in enumerate(fold_splits, start=1):
    build_one_fold_feature_cache(fold_number, fit_indices, valid_indices)

# Rebuild global diagnostic tables from saved per-fold artifacts; also works on resume.
feature_schema_by_fold = {}
all_validation_neighbor_diagnostics = []
all_test_neighbor_diagnostics = []
all_validation_clusters = []
all_gmm_metadata = []
gmm_status_rows = []
gmm_bic_rows = []
gmm_size_rows = []
gmm_restart_rows = []
for fold_number in range(1, N_SPLITS + 1):
    fold_cache_dir = CACHE_DIR / f"FOLD_{fold_number:02d}"
    schema_path = fold_cache_dir / "FEATURE_SCHEMA.json"
    feature_schema_by_fold[str(fold_number)] = json.loads(
        schema_path.read_text(encoding="utf-8")
    )
    all_validation_neighbor_diagnostics.append(
        pd.read_csv(fold_cache_dir / "VALIDATION_NEIGHBOR_DIAGNOSTICS.csv")
    )
    all_test_neighbor_diagnostics.append(
        pd.read_csv(fold_cache_dir / "TEST_NEIGHBOR_DIAGNOSTICS.csv")
    )
    all_validation_clusters.append(pd.read_csv(fold_cache_dir / "VALIDATION_CLUSTERS.csv"))

    gmm_fold_dir = GMM_DIR / f"FOLD_{fold_number:02d}"
    metadata = json.loads(
        (gmm_fold_dir / "segmenter_metadata.json").read_text(encoding="utf-8")
    )
    all_gmm_metadata.append(metadata)
    status_payload = json.loads(
        (gmm_fold_dir / "GMM_CACHE_STATUS.json").read_text(encoding="utf-8")
    )
    gmm_status_rows.append(status_payload)
    for branch_name, branch_metadata in [
        ("GOODS_SPIKE_ONLY_1D", metadata["goods_spike_only_1d_gmm"]),
        ("NEITHER_SPIKE_2D", metadata["neither_spike_2d_gmm"]),
    ]:
        for row in branch_metadata["bic_search"]:
            gmm_bic_rows.append({"FOLD": fold_number, "BRANCH": branch_name, **row})
        for row in branch_metadata["candidate_component_sizes"]:
            gmm_size_rows.append({"FOLD": fold_number, "BRANCH": branch_name, **row})
        for row in branch_metadata["initialization_stability"]["restart_details"]:
            gmm_restart_rows.append(
                {
                    "FOLD": fold_number,
                    "BRANCH": branch_name,
                    "SELECTED_K": branch_metadata["selected_n_components"],
                    **row,
                }
            )

atomic_json(OUTPUT_DIR / "feature_schema_by_fold.json", feature_schema_by_fold)
atomic_csv(
    OUTPUT_DIR / "validation_local500_neighbor_diagnostics_by_fold.csv",
    pd.concat(all_validation_neighbor_diagnostics, ignore_index=True),
)
atomic_csv(
    OUTPUT_DIR / "test_local500_neighbor_diagnostics_by_fold.csv",
    pd.concat(all_test_neighbor_diagnostics, ignore_index=True),
)
atomic_csv(
    OUTPUT_DIR / "validation_loan_structure_clusters_by_fold.csv",
    pd.concat(all_validation_clusters, ignore_index=True),
)
atomic_csv(GMM_DIR / "gmm_bic_by_k_all_folds.csv", pd.DataFrame(gmm_bic_rows))
atomic_csv(
    GMM_DIR / "gmm_candidate_component_sizes_by_k.csv",
    pd.DataFrame(gmm_size_rows),
)
atomic_csv(
    GMM_DIR / "gmm_initialization_stability_by_fold.csv",
    pd.DataFrame(gmm_restart_rows),
)
atomic_csv(GMM_DIR / "gmm_cache_status_by_fold.csv", pd.DataFrame(gmm_status_rows))
atomic_json(
    GMM_DIR / "gmm_cache_manifest.json",
    {
        "persistent_cache_directory": str(GMM_CACHE_DIR),
        "cache_schema_version": GMM_CACHE_SCHEMA_VERSION,
        "cache_input_columns": GMM_CACHE_INPUT_COLUMNS,
        "target_used_for_gmm_fitting": False,
        "folds": gmm_status_rows,
    },
)
atomic_json(
    GMM_DIR / "hybrid_gmm_segmentation_by_fold.json",
    {
        "gmm_max_components": GMM_MAX_COMPONENTS,
        "gmm_min_component_rows": GMM_MIN_COMPONENT_ROWS,
        "gmm_stability_repeats": GMM_STABILITY_REPEATS,
        "target_used": False,
        "folds": all_gmm_metadata,
    },
)


# =============================================================================
# 11. MODEL FACTORIES — CUDA / CPU DEVICE ALLOCATION
# =============================================================================


def make_catboost() -> CatBoostClassifier:
    return CatBoostClassifier(
        **FIXED_CATBOOST_PARAMS,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        random_seed=RANDOM_SEED,
        allow_writing_files=False,
        verbose=200,
        thread_count=CPU_THREADS,
    )


def make_lightgbm() -> LGBMClassifier:
    return LGBMClassifier(
        **FIXED_LIGHTGBM_PARAMS,
        objective="binary",
        metric="auc",
        device_type="cpu",
        random_state=RANDOM_SEED,
        n_jobs=CPU_THREADS,
        subsample_freq=1,
        verbosity=-1,
    )


def make_xgboost() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **FIXED_XGBOOST_PARAMS,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device="cuda",
        enable_categorical=True,
        random_state=RANDOM_SEED,
        n_jobs=CPU_THREADS,
        verbosity=0,
    )


# =============================================================================
# 13. OPTUNA 5-FOLD INNER CV + FINAL TRAINING FOR ALL FOUR MODELS
# =============================================================================

# Optuna is installed here as well as imported here so this replacement block
# remains self-contained when pasted into the original Colab script.
try:
    import optuna
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "optuna>=4.6"])
    import optuna

from optuna.trial import TrialState
from lightgbm import early_stopping as lgb_early_stopping
from lightgbm import log_evaluation as lgb_log_evaluation

# Required local GPU setup: RealMLP, CatBoost and XGBoost use CUDA; LightGBM stays CPU.
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for RealMLP-TD, CatBoost GPU and XGBoost GPU in this run.")
print(f"[GPU CHECK] {torch.cuda.get_device_name(0)} | CUDA={torch.version.cuda}", flush=True)

# ----------------------------- Optuna settings -----------------------------
REALMLP_INNER_VALIDATION_FRACTION = 0.20
REALMLP_INNER_SPLIT_SEED_OFFSET = 70000
OPTUNA_TRIALS_PER_MODEL_PER_FOLD = 12
OPTUNA_INNER_N_SPLITS = 5
OPTUNA_EARLY_STOPPING_ROUNDS = 200
OPTUNA_REALMLP_EPOCH_VALIDATION_FRACTION = 0.20
OPTUNA_RANDOM_SEED_OFFSET = 70000
OPTUNA_SEARCH_SPACE_VERSION = "four-model-5fold-gpu-career-profile-search-space-v4"
OPTUNA_N_JOBS = 1  # One study at a time; each estimator uses CPU_THREADS.

OPTUNA_DIR = OUTPUT_DIR / "OPTUNA"
OPTUNA_STUDY_DIR = OPTUNA_DIR / "STUDIES"
OPTUNA_TRIALS_DIR = OPTUNA_DIR / "TRIALS"
OPTUNA_SPLITS_DIR = OPTUNA_DIR / "INNER_SPLITS"
for _p in [OPTUNA_DIR, OPTUNA_STUDY_DIR, OPTUNA_TRIALS_DIR, OPTUNA_SPLITS_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

def _optuna_package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


OPTUNA_MODEL_CONFIG = {
    "base_preprocessing_config_sha256": CONFIG_HASH,
    "search_space_version": OPTUNA_SEARCH_SPACE_VERSION,
    "trials_per_model_per_outer_fold": OPTUNA_TRIALS_PER_MODEL_PER_FOLD,
    "inner_n_splits": OPTUNA_INNER_N_SPLITS,
    "realmlp_epoch_validation_fraction": OPTUNA_REALMLP_EPOCH_VALIDATION_FRACTION,
    "early_stopping_rounds": OPTUNA_EARLY_STOPPING_ROUNDS,
    "objective": "maximize_mean_inner_5fold_roc_auc",
    "outer_validation_used_for_tuning": False,
    "outer_folds": N_SPLITS,
    "models": ALL_MODEL_NAMES,
    "random_seed": RANDOM_SEED,
    "random_seed_offset": OPTUNA_RANDOM_SEED_OFFSET,
    "cpu_threads": CPU_THREADS,
    "hardware": {"realmlp": "cuda", "catboost": "gpu", "lightgbm": "cpu", "xgboost": "cuda"},
    "library_versions": {
        "optuna": getattr(optuna, "__version__", "unknown"),
        "catboost": _optuna_package_version("catboost"),
        "lightgbm": _optuna_package_version("lightgbm"),
        "xgboost": _optuna_package_version("xgboost"),
        "pytabkit": _optuna_package_version("pytabkit"),
    },
    "inner_cv": "StratifiedKFold(n_splits=5, shuffle=True), one objective score per fold averaged",
    "realmlp_epoch_selection": {
        "method": "a separate stratified split inside each inner-training partition, then refit on that inner-training partition",
        "final_refit": "all outer-training rows at epoch selected inside outer training",
        "outer_validation_used_for_epoch_selection": False,
    },
}
OPTUNA_MODEL_CONFIG_HASH = sha256_json(OPTUNA_MODEL_CONFIG)
atomic_json(
    OPTUNA_DIR / "optuna_config.json",
    {"config_sha256": OPTUNA_MODEL_CONFIG_HASH, "configuration": OPTUNA_MODEL_CONFIG},
)

OPTUNA_SUMMARY: dict[str, Any] = {
    "config_sha256": OPTUNA_MODEL_CONFIG_HASH,
    "objective": "ROC-AUC",
    "trials_per_model_per_outer_fold": OPTUNA_TRIALS_PER_MODEL_PER_FOLD,
    "inner_n_splits": OPTUNA_INNER_N_SPLITS,
    "realmlp_epoch_validation_fraction": OPTUNA_REALMLP_EPOCH_VALIDATION_FRACTION,
    "early_stopping_rounds": OPTUNA_EARLY_STOPPING_ROUNDS,
    "outer_validation_used_for_tuning": False,
    "per_fold": {},
}
OPTUNA_FOLD_MODEL_HASHES: dict[tuple[str, int], str] = {}


def _checkpoint_hash_for_model_fold(model_name: str, fold_number: int) -> str:
    key = (model_name, int(fold_number))
    if key not in OPTUNA_FOLD_MODEL_HASHES:
        raise RuntimeError(f"Optuna result hash was not initialized for {model_name}/fold {fold_number}")
    return OPTUNA_FOLD_MODEL_HASHES[key]


def _optuna_inner_folds(fold_number: int, y_fit: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create/reuse deterministic stratified 5-fold splits inside Outer Training."""
    y_fit = np.asarray(y_fit, dtype=np.int32)
    split_path = OPTUNA_SPLITS_DIR / f"FOLD_{fold_number:02d}_INNER_5FOLD.npz"
    seed = RANDOM_SEED + OPTUNA_RANDOM_SEED_OFFSET + fold_number
    if split_path.exists():
        with np.load(split_path, allow_pickle=False) as z:
            saved_seed = int(np.asarray(z["random_state"]).reshape(-1)[0])
            saved_n = int(np.asarray(z["n_splits"]).reshape(-1)[0])
            folds = [(np.asarray(z[f"train_{i}"], dtype=np.int64),
                      np.asarray(z[f"valid_{i}"], dtype=np.int64)) for i in range(OPTUNA_INNER_N_SPLITS)]
        if saved_seed != seed or saved_n != OPTUNA_INNER_N_SPLITS:
            raise RuntimeError(f"Fold {fold_number}: saved inner-fold settings differ; use a new OPTUNA_SEARCH_SPACE_VERSION or remove {split_path}")
    else:
        splitter = StratifiedKFold(n_splits=OPTUNA_INNER_N_SPLITS, shuffle=True, random_state=seed)
        folds = [(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
                 for a, b in splitter.split(np.zeros(len(y_fit), dtype=np.int8), y_fit)]
        payload = {f"train_{i}": a for i, (a, _) in enumerate(folds)}
        payload.update({f"valid_{i}": b for i, (_, b) in enumerate(folds)})
        payload["random_state"] = np.asarray([seed], dtype=np.int64)
        payload["n_splits"] = np.asarray([OPTUNA_INNER_N_SPLITS], dtype=np.int64)
        atomic_npz(split_path, **payload)
    coverage = np.zeros(len(y_fit), dtype=np.int8)
    for i, (tr, va) in enumerate(folds):
        if len(tr) + len(va) != len(y_fit) or np.intersect1d(tr, va).size:
            raise RuntimeError(f"Fold {fold_number}: invalid inner split {i+1}")
        if tr.min(initial=0) < 0 or va.min(initial=0) < 0 or tr.max(initial=-1) >= len(y_fit) or va.max(initial=-1) >= len(y_fit):
            raise RuntimeError(f"Fold {fold_number}: inner split {i+1} has out-of-range positions")
        if len(np.unique(y_fit[tr])) != 2 or len(np.unique(y_fit[va])) != 2:
            raise RuntimeError(f"Fold {fold_number}: inner split {i+1} lacks a class")
        coverage[va] += 1
    if not np.all(coverage == 1):
        raise RuntimeError(f"Fold {fold_number}: inner validation folds do not partition Outer Training exactly once")
    return folds


def _study_for_model_fold(model_name: str, fold_number: int):
    study_dir = OPTUNA_STUDY_DIR / f"FOLD_{fold_number:02d}"
    study_dir.mkdir(parents=True, exist_ok=True)
    db_path = study_dir / f"{model_name}.sqlite3"
    storage_url = f"sqlite:///{db_path.as_posix()}"
    study_name = (
        f"home_credit_{model_name}_fold_{fold_number}_{OPTUNA_SEARCH_SPACE_VERSION}"
        f"_inner{OPTUNA_INNER_N_SPLITS}fold"
        f"_es{OPTUNA_EARLY_STOPPING_ROUNDS}"
        f"_mlpepoch{int(round(OPTUNA_REALMLP_EPOCH_VALIDATION_FRACTION * 100))}"
        f"_seed{OPTUNA_RANDOM_SEED_OFFSET}"
    )
    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_SEED + OPTUNA_RANDOM_SEED_OFFSET + 100 * fold_number + ALL_MODEL_NAMES.index(model_name),
        n_startup_trials=min(5, OPTUNA_TRIALS_PER_MODEL_PER_FOLD),
    )
    return optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )


def _persist_study(study: Any, model_name: str, fold_number: int) -> dict[str, Any]:
    """Persist complete trial history and the selected parameters for audit/resume."""
    trial_dir = OPTUNA_TRIALS_DIR / f"FOLD_{fold_number:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    trials_path = trial_dir / f"{model_name}_trials.csv"
    best_path = trial_dir / f"{model_name}_best.json"

    trials_frame = study.trials_dataframe(
        attrs=("number", "value", "params", "state", "user_attrs", "datetime_start", "datetime_complete")
    )
    if isinstance(trials_frame.columns, pd.MultiIndex):
        trials_frame.columns = ["_".join(str(x) for x in col if str(x)) for col in trials_frame.columns]
    atomic_csv(trials_path, trials_frame)

    if not any(t.state == TrialState.COMPLETE for t in study.trials):
        raise RuntimeError(f"Optuna study {model_name}/fold {fold_number} has no completed trials")
    best_trial = study.best_trial
    payload = {
        "model": model_name,
        "fold": int(fold_number),
        "study_name": study.study_name,
        "completed_trials": int(sum(t.state == TrialState.COMPLETE for t in study.trials)),
        "total_trials": int(len(study.trials)),
        "best_trial_number": int(best_trial.number),
        "best_inner_roc_auc": float(best_trial.value),
        "best_parameters": study.best_params,
        "best_trial_user_attributes": best_trial.user_attrs,
        "objective": "maximize mean ROC-AUC across 5 stratified inner folds of Outer Training",
        "outer_validation_used_for_tuning": False,
        "trials_csv": str(trials_path),
    }
    atomic_json(best_path, payload)
    OPTUNA_SUMMARY["per_fold"].setdefault(str(fold_number), {})[model_name] = payload
    atomic_json(OPTUNA_DIR / "OPTUNA_SUMMARY.json", OPTUNA_SUMMARY)
    return payload


def _optimize_study(study: Any, objective: Any, model_name: str, fold_number: int) -> dict[str, Any]:
    completed = sum(t.state == TrialState.COMPLETE for t in study.trials)
    remaining = max(0, OPTUNA_TRIALS_PER_MODEL_PER_FOLD - completed)
    if remaining:
        print(
            f"[OPTUNA] {model_name.upper()} fold {fold_number}: "
            f"{completed}/{OPTUNA_TRIALS_PER_MODEL_PER_FOLD} complete; running {remaining} trials",
            flush=True,
        )
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=OPTUNA_N_JOBS,
            gc_after_trial=True,
            show_progress_bar=False,
        )
    else:
        print(f"[OPTUNA] {model_name.upper()} fold {fold_number}: restoring completed study.", flush=True)
    result = _persist_study(study, model_name, fold_number)
    # Bind each fitted model checkpoint to the exact selected trial. If a study
    # database is lost/recreated, a different best trial cannot reuse an old model.
    OPTUNA_FOLD_MODEL_HASHES[(model_name, int(fold_number))] = sha256_json({
        "global_optuna_config_sha256": OPTUNA_MODEL_CONFIG_HASH,
        "model": model_name,
        "fold": int(fold_number),
        "best_trial_number": result["best_trial_number"],
        "best_parameters": result["best_parameters"],
        "best_trial_user_attributes": result["best_trial_user_attributes"],
    })
    return result


def _suggest_catboost_params(trial: Any) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 1800, 5200, step=200),
        "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.08, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 40.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.05, 3.0, log=True),
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        "bootstrap_type": "Bayesian",
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
    }


def _suggest_lightgbm_params(trial: Any) -> dict[str, Any]:
    max_depth = trial.suggest_int("max_depth", 4, 10)
    return {
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int("num_leaves", 8, min(64, 2 ** max_depth)),
        "n_estimators": trial.suggest_int("n_estimators", 1600, 5200, step=200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.06, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 80, 500, step=20),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 1.0),
        "subsample": trial.suggest_float("subsample", 0.60, 1.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.5),
        "max_bin": trial.suggest_categorical("max_bin", [255, 511, 1023]),
        "cat_smooth": trial.suggest_float("cat_smooth", 5.0, 120.0),
        "cat_l2": trial.suggest_float("cat_l2", 0.1, 15.0, log=True),
    }


def _suggest_xgboost_params(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 1200, 4200, step=200),
        "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_child_weight": trial.suggest_float("min_child_weight", 3.0, 100.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.55, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 20.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [256, 512, 1024]),
        "max_cat_to_onehot": trial.suggest_int("max_cat_to_onehot", 4, 16),
    }


_REALMLP_ARCHITECTURES = {
    "128x2": [128, 128],
    "256x2": [256, 256],
    "256x3": [256, 256, 256],
    "512x256x128": [512, 256, 128],
}


def _suggest_realmlp_params(trial: Any) -> dict[str, Any]:
    architecture = trial.suggest_categorical("architecture", list(_REALMLP_ARCHITECTURES))
    return {
        "architecture": architecture,
        "hidden_sizes": list(_REALMLP_ARCHITECTURES[architecture]),
        "n_epochs": trial.suggest_categorical("n_epochs", [128, 192, 256, 320]),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "lr": trial.suggest_float("lr", 0.003, 0.05, log=True),
        "p_drop": trial.suggest_float("p_drop", 0.0, 0.40),
        "wd": trial.suggest_float("wd", 1e-6, 0.10, log=True),
    }


def _realmlp_constructor_params(sampled: dict[str, Any], fold_number: int) -> dict[str, Any]:
    params = dict(REALMLP_PARAMS)
    params.update({
        "device": DEVICE,
        "n_threads": CPU_THREADS,
        "hidden_sizes": list(sampled["hidden_sizes"]),
        "n_epochs": int(sampled["n_epochs"]),
        "batch_size": int(sampled["batch_size"]),
        "lr": float(sampled["lr"]),
        "p_drop": float(sampled["p_drop"]),
        "wd": float(sampled["wd"]),
        "random_state": int(RANDOM_SEED + OPTUNA_RANDOM_SEED_OFFSET + fold_number),
    })
    return params


def extract_realmlp_stop_epoch(model: Any, max_epochs: int | None = None) -> int:
    """Extract one scalar stop_epoch from a single-CV RealMLP fit."""
    fit_params = getattr(model, "fit_params_", None)
    if not isinstance(fit_params, dict) or "stop_epoch" not in fit_params:
        raise RuntimeError("RealMLP did not expose fit_params_['stop_epoch']")
    payload = fit_params["stop_epoch"]
    if isinstance(payload, dict):
        values = list(payload.values())
    elif isinstance(payload, (list, tuple, np.ndarray)):
        values = list(payload)
    else:
        values = [payload]
    if len(values) != 1:
        raise RuntimeError(f"Expected one RealMLP stop_epoch, received {len(values)}")
    selected = np.asarray(values[0]).reshape(-1)
    if selected.size != 1:
        raise RuntimeError(f"Expected scalar RealMLP stop_epoch, got shape {selected.shape}")
    epoch = int(selected[0])
    cap = int(max_epochs if max_epochs is not None else REALMLP_PARAMS["n_epochs"])
    if epoch < 1 or epoch > cap:
        raise RuntimeError(f"Invalid RealMLP stop_epoch={epoch}; expected 1..{cap}")
    return epoch


def _median_best_iterations(values: list[int]) -> int:
    return max(1, int(np.median(np.asarray(values, dtype=np.int64))))


def _tune_catboost(fold_number: int, cache: dict[str, Any], folds: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    X_raw = cache["tree_fit"]
    y_all = np.asarray(cache["y_fit"], dtype=np.int32)
    cat_cols = cache["tree_categorical_columns"]
    def objective(trial: Any) -> float:
        params = _suggest_catboost_params(trial)
        scores, aps, best_iterations = [], [], []
        for inner_i, (tr, va) in enumerate(folds, 1):
            profile_tr, profile_va = build_inner_career_profiles(X_raw, tr, va)
            Xtr_raw = X_raw.iloc[tr].copy()
            Xva_raw = X_raw.iloc[va].copy()
            Xtr_raw[CAREER_PROFILE_FEATURE] = profile_tr
            Xva_raw[CAREER_PROFILE_FEATURE] = profile_va
            Xtr = prepare_catboost_data(Xtr_raw, cat_cols)
            Xva = prepare_catboost_data(Xva_raw, cat_cols)
            model = CatBoostClassifier(**params, loss_function="Logloss", eval_metric="AUC", task_type="GPU",
                devices="0", random_seed=RANDOM_SEED + fold_number + inner_i, allow_writing_files=False,
                verbose=False, thread_count=CPU_THREADS)
            with threadpool_limits(limits=CPU_THREADS):
                model.fit(Xtr, y_all[tr], cat_features=cat_cols,
                    eval_set=(Xva, y_all[va]), early_stopping_rounds=OPTUNA_EARLY_STOPPING_ROUNDS,
                    use_best_model=True, verbose=False)
            bi = model.get_best_iteration()
            best_iterations.append(int(model.tree_count_ if bi is None or bi < 0 else bi + 1))
            pred = np.asarray(model.predict_proba(Xva)[:, 1], dtype=np.float64)
            scores.append(float(roc_auc_score(y_all[va], pred)))
            aps.append(float(average_precision_score(y_all[va], pred)))
            del model, Xtr_raw, Xva_raw, Xtr, Xva
            gc.collect()
        trial.set_user_attr("best_iterations_by_inner_fold", best_iterations)
        trial.set_user_attr("best_iterations", _median_best_iterations(best_iterations))
        trial.set_user_attr("inner_auc_by_fold", scores)
        trial.set_user_attr("inner_average_precision_by_fold", aps)
        return float(np.mean(scores))
    study = _study_for_model_fold("catboost", fold_number)
    return _optimize_study(study, objective, "catboost", fold_number)


def _tune_lightgbm(fold_number: int, cache: dict[str, Any], folds: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    X_raw = cache["tree_fit"]
    y_all = np.asarray(cache["y_fit"], dtype=np.int32)
    cat_cols = cache["tree_categorical_columns"]
    def objective(trial: Any) -> float:
        params = _suggest_lightgbm_params(trial)
        scores, aps, best_iterations = [], [], []
        for inner_i, (tr, va) in enumerate(folds, 1):
            profile_tr, profile_va = build_inner_career_profiles(X_raw, tr, va)
            Xtr_raw = X_raw.iloc[tr].copy()
            Xva_raw = X_raw.iloc[va].copy()
            Xtr_raw[CAREER_PROFILE_FEATURE] = profile_tr
            Xva_raw[CAREER_PROFILE_FEATURE] = profile_va
            Xtr, Xva, _ = prepare_gbdt_categorical_views(Xtr_raw, Xva_raw, Xva_raw, cat_cols)
            model = LGBMClassifier(**params, objective="binary", metric="auc", device_type="cpu",
                random_state=RANDOM_SEED + fold_number + inner_i, n_jobs=CPU_THREADS,
                subsample_freq=1, verbosity=-1)
            with threadpool_limits(limits=CPU_THREADS):
                model.fit(Xtr, y_all[tr], categorical_feature=cat_cols, eval_set=[(Xva, y_all[va])],
                    eval_metric="auc", callbacks=[lgb_early_stopping(OPTUNA_EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False), lgb_log_evaluation(period=0)])
            bi = int(getattr(model, "best_iteration_", 0) or params["n_estimators"])
            pred = np.asarray(model.predict_proba(Xva, num_iteration=bi)[:, 1], dtype=np.float64)
            scores.append(float(roc_auc_score(y_all[va], pred)))
            aps.append(float(average_precision_score(y_all[va], pred)))
            best_iterations.append(bi)
            del model, Xtr, Xva
            gc.collect()
        trial.set_user_attr("best_iterations_by_inner_fold", best_iterations)
        trial.set_user_attr("best_iterations", _median_best_iterations(best_iterations))
        trial.set_user_attr("inner_auc_by_fold", scores)
        trial.set_user_attr("inner_average_precision_by_fold", aps)
        return float(np.mean(scores))
    study = _study_for_model_fold("lightgbm", fold_number)
    return _optimize_study(study, objective, "lightgbm", fold_number)


def _tune_xgboost(fold_number: int, cache: dict[str, Any], folds: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    X_raw = cache["tree_fit"]
    y_all = np.asarray(cache["y_fit"], dtype=np.int32)
    cat_cols = cache["tree_categorical_columns"]
    def objective(trial: Any) -> float:
        params = _suggest_xgboost_params(trial)
        scores, aps, best_iterations = [], [], []
        for inner_i, (tr, va) in enumerate(folds, 1):
            profile_tr, profile_va = build_inner_career_profiles(X_raw, tr, va)
            Xtr_raw = X_raw.iloc[tr].copy()
            Xva_raw = X_raw.iloc[va].copy()
            Xtr_raw[CAREER_PROFILE_FEATURE] = profile_tr
            Xva_raw[CAREER_PROFILE_FEATURE] = profile_va
            Xtr, Xva, _ = prepare_gbdt_categorical_views(Xtr_raw, Xva_raw, Xva_raw, cat_cols)
            model = xgb.XGBClassifier(**params, objective="binary:logistic", eval_metric="auc",
                tree_method="hist", device="cuda", enable_categorical=True,
                random_state=RANDOM_SEED + fold_number + inner_i, n_jobs=CPU_THREADS,
                verbosity=0, early_stopping_rounds=OPTUNA_EARLY_STOPPING_ROUNDS)
            with threadpool_limits(limits=CPU_THREADS):
                model.fit(Xtr, y_all[tr], eval_set=[(Xva, y_all[va])], verbose=False)
            bir = getattr(model, "best_iteration", None)
            bi = int(params["n_estimators"] if bir is None else max(1, int(bir) + 1))
            pred = np.asarray(model.predict_proba(Xva, iteration_range=(0, bi))[:, 1], dtype=np.float64)
            scores.append(float(roc_auc_score(y_all[va], pred)))
            aps.append(float(average_precision_score(y_all[va], pred)))
            best_iterations.append(bi)
            del model, Xtr, Xva
            gc.collect()
        trial.set_user_attr("best_iterations_by_inner_fold", best_iterations)
        trial.set_user_attr("best_iterations", _median_best_iterations(best_iterations))
        trial.set_user_attr("inner_auc_by_fold", scores)
        trial.set_user_attr("inner_average_precision_by_fold", aps)
        return float(np.mean(scores))
    study = _study_for_model_fold("xgboost", fold_number)
    return _optimize_study(study, objective, "xgboost", fold_number)


def _tune_realmlp(fold_number: int, cache: dict[str, Any], folds: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    X_all = cache["mlp_fit"]
    X_raw = cache["tree_fit"]
    y_all = np.asarray(cache["y_fit"], dtype=np.int32)
    categorical_columns = cache["mlp_categorical_columns"]
    def objective(trial: Any) -> float:
        sampled = _suggest_realmlp_params(trial)
        params = _realmlp_constructor_params(sampled, fold_number)
        scores, aps, selected_epochs = [], [], []
        for inner_i, (tr, va) in enumerate(folds, 1):
            y_tr = y_all[tr]
            # Profile for the inner scorer is fitted on inner-training rows only.
            profile_tr, profile_va = build_inner_career_profiles(X_raw, tr, va)
            X_inner_train, X_inner_valid = replace_mlp_career_profile_codes(
                X_all.iloc[tr], X_all.iloc[va], profile_tr, profile_va
            )

            # The epoch-selector split is another validation boundary: refit the
            # hierarchical category counts on its epoch-training subset only.
            epoch_splitter = StratifiedShuffleSplit(n_splits=1, test_size=OPTUNA_REALMLP_EPOCH_VALIDATION_FRACTION,
                random_state=RANDOM_SEED + OPTUNA_RANDOM_SEED_OFFSET + fold_number * 100 + inner_i)
            ep_tr_rel, ep_va_rel = next(epoch_splitter.split(np.zeros(len(y_tr), dtype=np.int8), y_tr))
            ep_tr_pos = tr[ep_tr_rel]
            ep_va_pos = tr[ep_va_rel]
            profile_ep_tr = build_career_profile(X_raw.iloc[ep_tr_pos], X_raw.iloc[ep_tr_pos])
            profile_ep_va = build_career_profile(X_raw.iloc[ep_tr_pos], X_raw.iloc[ep_va_pos])
            X_epoch_train, X_epoch_valid = replace_mlp_career_profile_codes(
                X_all.iloc[ep_tr_pos], X_all.iloc[ep_va_pos], profile_ep_tr, profile_ep_va
            )

            selector_params = dict(params)
            selector_params["random_state"] = int(params["random_state"] + inner_i)
            selector = RealMLP_TD_Classifier(**selector_params)
            with threadpool_limits(limits=CPU_THREADS):
                selector.fit(X_epoch_train, y_tr[ep_tr_rel], X_val=X_epoch_valid,
                    y_val=y_tr[ep_va_rel], cat_col_names=categorical_columns)
            selected_epoch = extract_realmlp_stop_epoch(selector, max_epochs=params["n_epochs"])
            del selector, X_epoch_train, X_epoch_valid
            gc.collect()
            scoring_params = dict(params)
            scoring_params.update({"stop_epoch": selected_epoch, "val_fraction": 0.0, "n_refit": 0,
                                   "random_state": int(params["random_state"] + 1000 + inner_i)})
            scorer = RealMLP_TD_Classifier(**scoring_params)
            with threadpool_limits(limits=CPU_THREADS):
                scorer.fit(X_inner_train, y_tr, cat_col_names=categorical_columns)
            pred = positive_class_probability(scorer, X_inner_valid)
            scores.append(float(roc_auc_score(y_all[va], pred)))
            aps.append(float(average_precision_score(y_all[va], pred)))
            selected_epochs.append(int(selected_epoch))
            del scorer, X_inner_train, X_inner_valid
            gc.collect()
            if "torch" in globals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
        trial.set_user_attr("selected_epochs_by_inner_fold", selected_epochs)
        trial.set_user_attr("inner_auc_by_fold", scores)
        trial.set_user_attr("inner_average_precision_by_fold", aps)
        trial.set_user_attr("selected_epoch_median", int(np.median(selected_epochs)))
        return float(np.mean(scores))
    study = _study_for_model_fold("realmlp", fold_number)
    return _optimize_study(study, objective, "realmlp", fold_number)


def _tune_one_model(model_name: str, fold_number: int, cache: dict[str, Any],
                    folds: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    tuners = {
        "realmlp": _tune_realmlp,
        "catboost": _tune_catboost,
        "lightgbm": _tune_lightgbm,
        "xgboost": _tune_xgboost,
    }
    result = tuners[model_name](fold_number, cache, folds)
    print(
        f"[OPTUNA BEST] {model_name.upper()} fold {fold_number}: "
        f"inner AUC={result['best_inner_roc_auc']:.7f} | "
        f"completed trials={result['completed_trials']} | params={result['best_parameters']}",
        flush=True,
    )
    return result


# ------------------------ Model/fold checkpoint I/O -------------------------
def model_fold_directory(model_name: str, fold_number: int) -> Path:
    path = MODEL_DIR / model_name.upper() / f"FOLD_{fold_number:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def prediction_fold_directory(model_name: str, fold_number: int) -> Path:
    path = PREDICTION_DIR / model_name.upper() / f"FOLD_{fold_number:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model_fold_result(
    model_name: str, fold_number: int, model: Any, model_file: Path, model_kind: str,
    valid_indices: np.ndarray, valid_prediction: np.ndarray, test_prediction: np.ndarray,
    metrics: dict[str, float], importance_frame: pd.DataFrame | None = None,
    training_details: dict[str, Any] | None = None,
) -> None:
    model_fold_directory(model_name, fold_number)
    pred_dir = prediction_fold_directory(model_name, fold_number)
    atomic_model_save(model, model_file, model_kind)
    pred_path = pred_dir / "PREDICTIONS.npz"
    atomic_npz(
        pred_path,
        valid_indices=np.asarray(valid_indices, dtype=np.int64),
        valid_prediction=np.asarray(valid_prediction, dtype=np.float64),
        test_prediction=np.asarray(test_prediction, dtype=np.float64),
    )
    metrics_path = pred_dir / "METRICS.json"
    payload = {
        "model": model_name, "fold": fold_number,
        "validation_metrics": metrics,
        "valid_rows": int(len(valid_indices)),
        "test_rows": int(len(test_prediction)),
        "tuning_config_sha256": OPTUNA_MODEL_CONFIG_HASH,
    }
    if training_details is not None:
        payload["training_details"] = training_details
    atomic_json(metrics_path, payload)
    artifacts = [model_file, pred_path, metrics_path]
    if importance_frame is not None:
        importance_path = IMPORTANCE_DIR / model_name.upper() / f"FOLD_{fold_number:02d}_IMPORTANCE.csv"
        atomic_csv(importance_path, importance_frame)
        artifacts.append(importance_path)
    marker = pred_dir / "MODEL_FOLD_COMPLETE.json"
    mark_complete(marker, f"{model_name}_fold", _checkpoint_hash_for_model_fold(model_name, fold_number), artifacts)


def load_model_fold_result(model_name: str, fold_number: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pred_dir = prediction_fold_directory(model_name, fold_number)
    marker = pred_dir / "MODEL_FOLD_COMPLETE.json"
    pred_path = pred_dir / "PREDICTIONS.npz"
    metrics_path = pred_dir / "METRICS.json"
    if not checkpoint_is_complete(marker, f"{model_name}_fold", _checkpoint_hash_for_model_fold(model_name, fold_number)):
        raise RuntimeError(f"{model_name} fold {fold_number}: tuned checkpoint is incomplete")
    if not pred_path.exists() or not metrics_path.exists():
        raise RuntimeError(f"{model_name} fold {fold_number}: tuned checkpoint artifacts are missing")
    with np.load(pred_path, allow_pickle=False) as z:
        valid_prediction = np.asarray(z["valid_prediction"], dtype=np.float64)
        test_prediction = np.asarray(z["test_prediction"], dtype=np.float64)
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return valid_prediction, test_prediction, metrics_payload["validation_metrics"]


def _load_cached_tuning_result(model_name: str, fold_number: int) -> dict[str, Any]:
    path = OPTUNA_TRIALS_DIR / f"FOLD_{fold_number:02d}" / f"{model_name}_best.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Optuna best-parameter file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# -------------------------------- CatBoost ----------------------------------
def run_catboost_fold(fold_number: int, cache: dict[str, Any]):
    y_fit = np.asarray(cache["y_fit"], dtype=np.int32)
    inner_folds = _optuna_inner_folds(fold_number, y_fit)
    tuning = _tune_one_model("catboost", fold_number, cache, inner_folds)
    pred_dir = prediction_fold_directory("catboost", fold_number)
    marker = pred_dir / "MODEL_FOLD_COMPLETE.json"
    if checkpoint_is_complete(marker, "catboost_fold", _checkpoint_hash_for_model_fold("catboost", fold_number)):
        print(f"CatBoost fold {fold_number}: restoring Optuna-tuned checkpoint.", flush=True)
        return load_model_fold_result("catboost", fold_number)

    params = dict(tuning["best_parameters"])
    params["iterations"] = int(tuning["best_trial_user_attributes"].get("best_iterations", params["iterations"]))
    X_fit = prepare_catboost_data(cache["tree_fit"], cache["tree_categorical_columns"])
    X_valid = prepare_catboost_data(cache["tree_valid"], cache["tree_categorical_columns"])
    X_test_fold = prepare_catboost_data(cache["tree_test"], cache["tree_categorical_columns"])
    model = CatBoostClassifier(
        **params, loss_function="Logloss", eval_metric="AUC", task_type="GPU", devices="0",
        random_seed=RANDOM_SEED + fold_number, allow_writing_files=False,
        verbose=200, thread_count=CPU_THREADS,
    )
    with threadpool_limits(limits=CPU_THREADS):
        model.fit(X_fit, y_fit, cat_features=cache["tree_categorical_columns"], verbose=200)
    valid_prediction = np.asarray(model.predict_proba(X_valid)[:, 1], dtype=np.float64)
    test_prediction = np.asarray(model.predict_proba(X_test_fold)[:, 1], dtype=np.float64)
    check_probabilities(f"CatBoost fold {fold_number} validation", valid_prediction)
    check_probabilities(f"CatBoost fold {fold_number} test", test_prediction)
    metrics = calculate_metrics(cache["y_valid"], valid_prediction)
    importance = pd.DataFrame({
        "FEATURE": list(cache["tree_feature_names"]),
        "IMPORTANCE": model.get_feature_importance(type="PredictionValuesChange"),
    }).sort_values("IMPORTANCE", ascending=False)
    details = {
        "optuna_best_inner_roc_auc": tuning["best_inner_roc_auc"],
        "optuna_best_parameters": params,
        "final_iterations": params["iterations"],
        "outer_validation_used_for_tuning": False,
    }
    save_model_fold_result(
        "catboost", fold_number, model,
        model_fold_directory("catboost", fold_number) / "catboost_model.cbm", "catboost",
        cache["valid_indices"], valid_prediction, test_prediction, metrics, importance, details,
    )
    del model, X_fit, X_valid, X_test_fold
    gc.collect()
    return valid_prediction, test_prediction, metrics


# -------------------------------- LightGBM ----------------------------------
def run_lightgbm_fold(fold_number: int, cache: dict[str, Any]):
    y_fit = np.asarray(cache["y_fit"], dtype=np.int32)
    inner_folds = _optuna_inner_folds(fold_number, y_fit)
    tuning = _tune_one_model("lightgbm", fold_number, cache, inner_folds)
    pred_dir = prediction_fold_directory("lightgbm", fold_number)
    marker = pred_dir / "MODEL_FOLD_COMPLETE.json"
    if checkpoint_is_complete(marker, "lightgbm_fold", _checkpoint_hash_for_model_fold("lightgbm", fold_number)):
        print(f"LightGBM fold {fold_number}: restoring Optuna-tuned checkpoint.", flush=True)
        return load_model_fold_result("lightgbm", fold_number)

    params = dict(tuning["best_parameters"])
    params["n_estimators"] = int(tuning["best_trial_user_attributes"].get("best_iterations", params["n_estimators"]))
    X_fit, X_valid, X_test_fold = prepare_gbdt_categorical_views(
        cache["tree_fit"], cache["tree_valid"], cache["tree_test"], cache["tree_categorical_columns"]
    )
    model = LGBMClassifier(
        **params, objective="binary", metric="auc", device_type="cpu",
        random_state=RANDOM_SEED + fold_number, n_jobs=CPU_THREADS,
        subsample_freq=1, verbosity=-1,
    )
    with threadpool_limits(limits=CPU_THREADS):
        model.fit(X_fit, y_fit, categorical_feature=cache["tree_categorical_columns"])
    valid_prediction = np.asarray(model.predict_proba(X_valid)[:, 1], dtype=np.float64)
    test_prediction = np.asarray(model.predict_proba(X_test_fold)[:, 1], dtype=np.float64)
    check_probabilities(f"LightGBM fold {fold_number} validation", valid_prediction)
    check_probabilities(f"LightGBM fold {fold_number} test", test_prediction)
    metrics = calculate_metrics(cache["y_valid"], valid_prediction)
    importance = pd.DataFrame({
        "FEATURE": list(cache["tree_feature_names"]),
        "IMPORTANCE_GAIN": model.booster_.feature_importance(importance_type="gain"),
        "IMPORTANCE_SPLIT": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("IMPORTANCE_GAIN", ascending=False)
    details = {
        "optuna_best_inner_roc_auc": tuning["best_inner_roc_auc"],
        "optuna_best_parameters": params,
        "final_iterations": params["n_estimators"],
        "outer_validation_used_for_tuning": False,
    }
    save_model_fold_result(
        "lightgbm", fold_number, model,
        model_fold_directory("lightgbm", fold_number) / "lightgbm_model.txt", "lightgbm",
        cache["valid_indices"], valid_prediction, test_prediction, metrics, importance, details,
    )
    del model, X_fit, X_valid, X_test_fold
    gc.collect()
    return valid_prediction, test_prediction, metrics


# -------------------------------- XGBoost -----------------------------------
def run_xgboost_fold(fold_number: int, cache: dict[str, Any]):
    y_fit = np.asarray(cache["y_fit"], dtype=np.int32)
    inner_folds = _optuna_inner_folds(fold_number, y_fit)
    tuning = _tune_one_model("xgboost", fold_number, cache, inner_folds)
    pred_dir = prediction_fold_directory("xgboost", fold_number)
    marker = pred_dir / "MODEL_FOLD_COMPLETE.json"
    if checkpoint_is_complete(marker, "xgboost_fold", _checkpoint_hash_for_model_fold("xgboost", fold_number)):
        print(f"XGBoost fold {fold_number}: restoring Optuna-tuned checkpoint.", flush=True)
        return load_model_fold_result("xgboost", fold_number)

    params = dict(tuning["best_parameters"])
    params["n_estimators"] = int(tuning["best_trial_user_attributes"].get("best_iterations", params["n_estimators"]))
    X_fit, X_valid, X_test_fold = prepare_gbdt_categorical_views(
        cache["tree_fit"], cache["tree_valid"], cache["tree_test"], cache["tree_categorical_columns"]
    )
    model = xgb.XGBClassifier(
        **params, objective="binary:logistic", eval_metric="auc",
        tree_method="hist", device="cuda", enable_categorical=True,
        random_state=RANDOM_SEED + fold_number, n_jobs=CPU_THREADS, verbosity=0,
    )
    with threadpool_limits(limits=CPU_THREADS):
        model.fit(X_fit, y_fit, verbose=False)
    valid_prediction = np.asarray(model.predict_proba(X_valid)[:, 1], dtype=np.float64)
    test_prediction = np.asarray(model.predict_proba(X_test_fold)[:, 1], dtype=np.float64)
    check_probabilities(f"XGBoost fold {fold_number} validation", valid_prediction)
    check_probabilities(f"XGBoost fold {fold_number} test", test_prediction)
    metrics = calculate_metrics(cache["y_valid"], valid_prediction)
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    names = list(cache["tree_feature_names"])
    importance = pd.DataFrame({
        "FEATURE": names,
        "IMPORTANCE_GAIN": [gain.get(name, 0.0) for name in names],
        "IMPORTANCE_WEIGHT": [weight.get(name, 0.0) for name in names],
    }).sort_values("IMPORTANCE_GAIN", ascending=False)
    details = {
        "optuna_best_inner_roc_auc": tuning["best_inner_roc_auc"],
        "optuna_best_parameters": params,
        "final_iterations": params["n_estimators"],
        "outer_validation_used_for_tuning": False,
    }
    save_model_fold_result(
        "xgboost", fold_number, model,
        model_fold_directory("xgboost", fold_number) / "xgboost_model.json", "xgboost",
        cache["valid_indices"], valid_prediction, test_prediction, metrics, importance, details,
    )
    del model, booster, X_fit, X_valid, X_test_fold
    gc.collect()
    return valid_prediction, test_prediction, metrics


# -------------------------------- RealMLP -----------------------------------
def run_realmlp_fold(fold_number: int, cache: dict[str, Any]):
    y_fit = np.asarray(cache["y_fit"], dtype=np.int32)
    inner_folds = _optuna_inner_folds(fold_number, y_fit)
    tuning = _tune_one_model("realmlp", fold_number, cache, inner_folds)
    pred_dir = prediction_fold_directory("realmlp", fold_number)
    marker = pred_dir / "MODEL_FOLD_COMPLETE.json"
    if checkpoint_is_complete(marker, "realmlp_fold", _checkpoint_hash_for_model_fold("realmlp", fold_number)):
        print(f"RealMLP fold {fold_number}: restoring Optuna-tuned leakage-safe checkpoint.", flush=True)
        return load_model_fold_result("realmlp", fold_number)

    best_sampled = dict(tuning["best_parameters"])
    best_sampled["hidden_sizes"] = list(_REALMLP_ARCHITECTURES[best_sampled["architecture"]])
    refit_params = _realmlp_constructor_params(best_sampled, fold_number)

    # Select the final epoch on a split of all Outer-Training rows only.
    final_epoch_seed = RANDOM_SEED + REALMLP_INNER_SPLIT_SEED_OFFSET + fold_number
    epoch_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=REALMLP_INNER_VALIDATION_FRACTION,
        random_state=final_epoch_seed,
    )
    epoch_train_pos, epoch_valid_pos = next(
        epoch_splitter.split(np.zeros(len(y_fit), dtype=np.int8), y_fit)
    )
    X_fit = cache["mlp_fit"]
    X_valid = cache["mlp_valid"]
    X_test_fold = cache["mlp_test"]
    categorical_columns = cache["mlp_categorical_columns"]

    profile_epoch_train = build_career_profile(
        cache["tree_fit"].iloc[epoch_train_pos], cache["tree_fit"].iloc[epoch_train_pos]
    )
    profile_epoch_valid = build_career_profile(
        cache["tree_fit"].iloc[epoch_train_pos], cache["tree_fit"].iloc[epoch_valid_pos]
    )
    X_epoch_train, X_epoch_valid = replace_mlp_career_profile_codes(
        X_fit.iloc[epoch_train_pos], X_fit.iloc[epoch_valid_pos],
        profile_epoch_train, profile_epoch_valid,
    )
    selector = RealMLP_TD_Classifier(**refit_params)
    with threadpool_limits(limits=CPU_THREADS):
        selector.fit(
            X_epoch_train, y_fit[epoch_train_pos],
            X_val=X_epoch_valid, y_val=y_fit[epoch_valid_pos],
            cat_col_names=categorical_columns,
        )
    del X_epoch_train, X_epoch_valid
    selected_epoch = extract_realmlp_stop_epoch(selector, max_epochs=refit_params["n_epochs"])
    del selector
    gc.collect()

    final_params = dict(refit_params)
    final_params["stop_epoch"] = selected_epoch
    final_params["val_fraction"] = 0.0
    final_params["n_refit"] = 0
    model = RealMLP_TD_Classifier(**final_params)
    with threadpool_limits(limits=CPU_THREADS):
        model.fit(X_fit, y_fit, cat_col_names=categorical_columns)

    # Outer Validation is not supplied to either Optuna or epoch selection.
    valid_prediction = positive_class_probability(model, X_valid)
    test_prediction = positive_class_probability(model, X_test_fold)
    check_probabilities(f"RealMLP fold {fold_number} validation", valid_prediction)
    check_probabilities(f"RealMLP fold {fold_number} test", test_prediction)
    metrics = calculate_metrics(cache["y_valid"], valid_prediction)
    details = {
        "optuna_best_inner_roc_auc": tuning["best_inner_roc_auc"],
        "optuna_best_parameters": tuning["best_parameters"],
        "selected_stop_epoch": int(selected_epoch),
        "epoch_selection_method": "inner stratified split of all Outer Training rows",
        "epoch_selection_validation_rows": int(len(epoch_valid_pos)),
        "outer_train_rows_used_for_final_refit": int(len(y_fit)),
        "outer_validation_rows_used_for_epoch_selection": 0,
        "final_refit_val_fraction": 0.0,
        "final_refit_used_all_outer_training_rows": True,
        "outer_validation_used_only_for_final_evaluation": True,
    }
    print(
        f"RealMLP fold {fold_number}: selected_epoch={selected_epoch} | "
        f"Optuna inner AUC={tuning['best_inner_roc_auc']:.7f} | "
        f"Outer AUC={metrics['roc_auc']:.7f} | AP={metrics['average_precision']:.7f}",
        flush=True,
    )
    save_model_fold_result(
        "realmlp", fold_number, model,
        model_fold_directory("realmlp", fold_number) / "realmlp_model.pkl", "pickle",
        cache["valid_indices"], valid_prediction, test_prediction, metrics, None, details,
    )
    del model, X_fit, X_valid, X_test_fold
    gc.collect()
    return valid_prediction, test_prediction, metrics


# ----------------------- Tune/train every model/fold ------------------------
print_section("OPTUNA TUNING — ALL FOUR MODELS, STRICTLY INSIDE OUTER TRAINING")
all_fold_predictions: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
    name: {} for name in ALL_MODEL_NAMES
}
all_fold_metrics: dict[str, dict[int, dict[str, float]]] = {name: {} for name in ALL_MODEL_NAMES}

for fold_number in range(1, N_SPLITS + 1):
    print_section(f"OPTUNA + FINAL MODEL TRAINING — FOLD {fold_number}/{N_SPLITS}")
    cache = load_feature_cache(fold_number)
    for model_name, runner in [
        ("catboost", run_catboost_fold),
        ("lightgbm", run_lightgbm_fold),
        ("xgboost", run_xgboost_fold),
        ("realmlp", run_realmlp_fold),
    ]:
        start = time.time()
        print(f"Starting {model_name.upper()} fold {fold_number}...", flush=True)
        valid_pred, test_pred, metrics = runner(fold_number, cache)
        all_fold_predictions[model_name][fold_number] = (valid_pred, test_pred)
        all_fold_metrics[model_name][fold_number] = metrics
        elapsed = (time.time() - start) / 60.0
        print(
            f"{model_name.upper()} fold {fold_number} complete | "
            f"Outer AUC={metrics['roc_auc']:.7f} | AP={metrics['average_precision']:.7f} "
            f"| elapsed={elapsed:.2f} min",
            flush=True,
        )
        progress_rows = []
        for current_model in ALL_MODEL_NAMES:
            for current_fold, current_metrics in sorted(all_fold_metrics[current_model].items()):
                progress_rows.append({"MODEL": current_model, "FOLD": current_fold, **current_metrics})
        atomic_csv(OUTPUT_DIR / "model_fold_metrics_progress.csv", pd.DataFrame(progress_rows))
        gc.collect()
    del cache
    gc.collect()

# Ensure the study summary is complete after restoring model checkpoints too.
for fold_number in range(1, N_SPLITS + 1):
    for model_name in ALL_MODEL_NAMES:
        best_path = OPTUNA_TRIALS_DIR / f"FOLD_{fold_number:02d}" / f"{model_name}_best.json"
        if not best_path.exists():
            raise FileNotFoundError(f"Missing Optuna result: {best_path}")
        OPTUNA_SUMMARY["per_fold"].setdefault(str(fold_number), {})[model_name] = json.loads(
            best_path.read_text(encoding="utf-8")
        )
atomic_json(OPTUNA_DIR / "OPTUNA_SUMMARY.json", OPTUNA_SUMMARY)

# Persist the selected RealMLP epoch and best parameters for all five outer folds.
realmlp_epoch_rows = []
for fold_number in range(1, N_SPLITS + 1):
    metrics_path = prediction_fold_directory("realmlp", fold_number) / "METRICS.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    details = payload.get("training_details", {})
    if "selected_stop_epoch" not in details:
        raise RuntimeError(f"RealMLP fold {fold_number}: missing selected_stop_epoch metadata")
    realmlp_epoch_rows.append({"FOLD": fold_number, **details})
atomic_csv(OUTPUT_DIR / "realmlp_epoch_selection_by_fold.csv", pd.DataFrame(realmlp_epoch_rows))

print_section("OPTUNA FOUR-MODEL TRAINING COMPLETED — OUTER VALIDATION REMAINED UNTOUCHED")
print(f"Optuna studies directory : {OPTUNA_STUDY_DIR}")
print(f"Optuna trial reports     : {OPTUNA_TRIALS_DIR}")
print(f"Optuna summary           : {OPTUNA_DIR / 'OPTUNA_SUMMARY.json'}")
print(f"Model checkpoint hash    : {OPTUNA_MODEL_CONFIG_HASH}")

# =============================================================================
# 14. RECONSTRUCT COMPLETE OOF / TEST MATRICES FROM CHECKPOINTS
# =============================================================================

print_section("RECONSTRUCTING OOF / TEST PREDICTIONS FROM MODEL/FOLD CHECKPOINTS")
oof_predictions: dict[str, np.ndarray] = {}
test_fold_predictions: dict[str, np.ndarray] = {}
fold_metric_rows = []

for model_name in ALL_MODEL_NAMES:
    oof = np.full(n_train, np.nan, dtype=np.float64)
    test_by_fold = np.full((N_SPLITS, n_test), np.nan, dtype=np.float64)
    for fold_number in range(1, N_SPLITS + 1):
        valid_prediction, test_prediction, metrics = load_model_fold_result(
            model_name, fold_number
        )
        cache = load_feature_cache(fold_number)
        valid_indices = cache["valid_indices"]
        if len(valid_prediction) != len(valid_indices) or len(test_prediction) != n_test:
            raise RuntimeError(
                f"{model_name} fold {fold_number}: prediction dimensions do not match"
            )
        if np.isfinite(oof[valid_indices]).any():
            raise RuntimeError(f"{model_name}: overlapping OOF assignments in fold {fold_number}")
        oof[valid_indices] = valid_prediction
        test_by_fold[fold_number - 1] = test_prediction
        fold_metric_rows.append(
            {"MODEL": model_name, "FOLD": fold_number, **metrics}
        )
        del cache
    if not np.isfinite(oof).all():
        raise RuntimeError(f"{model_name}: OOF prediction coverage incomplete")
    if not np.isfinite(test_by_fold).all():
        raise RuntimeError(f"{model_name}: test fold predictions incomplete")
    check_probabilities(f"{model_name} OOF", oof)
    for fold_index in range(N_SPLITS):
        check_probabilities(f"{model_name} test fold {fold_index + 1}", test_by_fold[fold_index])
    oof_predictions[model_name] = oof
    test_fold_predictions[model_name] = test_by_fold

atomic_csv(OUTPUT_DIR / "model_fold_metrics_all.csv", pd.DataFrame(fold_metric_rows))


# =============================================================================
# 15. EQUAL-WEIGHT ENSEMBLE, FOLD METRICS, OOF, AND SUBMISSIONS
# =============================================================================

print_section("FINAL FOUR-MODEL EQUAL-WEIGHT ENSEMBLE")
if not np.isclose(sum(ENSEMBLE_WEIGHTS.values()), 1.0, atol=1e-12):
    raise RuntimeError("Ensemble weights do not sum to one")
ensemble_oof = sum(
    ENSEMBLE_WEIGHTS[name] * oof_predictions[name] for name in ALL_MODEL_NAMES
)
check_probabilities("Four-model ensemble OOF", ensemble_oof)
model_test_mean = {
    name: test_fold_predictions[name].mean(axis=0) for name in ALL_MODEL_NAMES
}
ensemble_test = sum(
    ENSEMBLE_WEIGHTS[name] * model_test_mean[name] for name in ALL_MODEL_NAMES
)
check_probabilities("Four-model ensemble test", ensemble_test)

fold_ensemble_rows = []
for fold_number, (_, valid_indices) in enumerate(fold_splits, start=1):
    y_valid = y[valid_indices]
    row = {
        "FOLD": fold_number,
        "N_VALID": int(len(valid_indices)),
        "N_POSITIVE": int(y_valid.sum()),
        "N_NEGATIVE": int((y_valid == 0).sum()),
    }
    for model_name in ALL_MODEL_NAMES:
        metrics = calculate_metrics(y_valid, oof_predictions[model_name][valid_indices])
        for metric_name, value in metrics.items():
            row[f"{model_name.upper()}_{metric_name.upper()}"] = value
    ensemble_metrics = calculate_metrics(y_valid, ensemble_oof[valid_indices])
    for metric_name, value in ensemble_metrics.items():
        row[f"ENSEMBLE_{metric_name.upper()}"] = value
    fold_ensemble_rows.append(row)

fold_metrics_df = pd.DataFrame(fold_ensemble_rows)
atomic_csv(
    ENSEMBLE_DIR / "four_model_equal_weight_five_fold_metrics.csv",
    fold_metrics_df,
)

pair_counts = np.asarray(
    [
        int(y[valid_indices].sum()) * int((y[valid_indices] == 0).sum())
        for _, valid_indices in fold_splits
    ],
    dtype=np.float64,
)
fold_auc = fold_metrics_df["ENSEMBLE_ROC_AUC"].to_numpy(dtype=np.float64)
pair_weighted_auc = float(np.average(fold_auc, weights=pair_counts))
pooled_metrics = calculate_metrics(y, ensemble_oof)
model_oof_metrics = {
    name: calculate_metrics(y, oof_predictions[name]) for name in ALL_MODEL_NAMES
}

# Save OOF predictions.
oof_frame = pd.DataFrame({ID_COL: train_ids, TARGET_COL: y})
for name in ALL_MODEL_NAMES:
    oof_frame[f"{name.upper()}_OOF_PROBABILITY"] = oof_predictions[name]
oof_frame["FOUR_MODEL_EQUAL_WEIGHT_ENSEMBLE_OOF"] = ensemble_oof
atomic_csv(ENSEMBLE_DIR / "oof_predictions_four_models.csv", oof_frame)

# Save every fold's test predictions and each model's five-fold mean.
test_frame = pd.DataFrame({ID_COL: test_ids})
for name in ALL_MODEL_NAMES:
    for fold_index in range(N_SPLITS):
        test_frame[f"{name.upper()}_FOLD_{fold_index + 1}"] = test_fold_predictions[name][fold_index]
    test_frame[f"{name.upper()}_MEAN_5FOLD"] = model_test_mean[name]
test_frame["FOUR_MODEL_EQUAL_WEIGHT_ENSEMBLE"] = ensemble_test
atomic_csv(ENSEMBLE_DIR / "test_predictions_all_folds_and_models.csv", test_frame)


def write_submission(
    probabilities: np.ndarray,
    path: Path,
    label: str,
) -> None:
    check_probabilities(label, probabilities)
    if len(probabilities) != len(test_ids):
        raise ValueError(f"{label}: test prediction count mismatch")
    prediction_map = pd.Series(probabilities, index=test_ids, name=TARGET_COL)
    result = sample_sub.copy()
    result[TARGET_COL] = result[ID_COL].map(prediction_map)
    if result[TARGET_COL].isna().any():
        raise RuntimeError(f"{label}: missing submission predictions after ID mapping")
    result = result.loc[:, sample_sub.columns]
    temporary = _tmp_path(path)
    result.to_csv(temporary, index=False, float_format="%.12g")
    os.replace(temporary, path)


for name in ALL_MODEL_NAMES:
    write_submission(
        model_test_mean[name],
        ENSEMBLE_DIR / f"submission_{name}_5fold.csv",
        name,
    )
write_submission(
    ensemble_test,
    ENSEMBLE_DIR / "submission_four_model_equal_weight.csv",
    "four-model equal-weight ensemble",
)

# Aggregate feature importances for the three tree models.
for model_name, value_columns in [
    ("catboost", ["IMPORTANCE"]),
    ("lightgbm", ["IMPORTANCE_GAIN", "IMPORTANCE_SPLIT"]),
    ("xgboost", ["IMPORTANCE_GAIN", "IMPORTANCE_WEIGHT"]),
]:
    importance_frames = []
    for fold_number in range(1, N_SPLITS + 1):
        importance_path = (
            IMPORTANCE_DIR
            / model_name.upper()
            / f"FOLD_{fold_number:02d}_IMPORTANCE.csv"
        )
        if importance_path.exists():
            frame = pd.read_csv(importance_path)
            frame["FOLD"] = fold_number
            importance_frames.append(frame)
    if importance_frames:
        combined = pd.concat(importance_frames, ignore_index=True)
        atomic_csv(
            IMPORTANCE_DIR / f"{model_name.upper()}_IMPORTANCE_BY_FOLD.csv",
            combined,
        )
        aggregated = combined.groupby("FEATURE", as_index=False)[value_columns].mean()
        aggregated = aggregated.sort_values(value_columns[0], ascending=False).reset_index(drop=True)
        atomic_csv(
            IMPORTANCE_DIR / f"{model_name.upper()}_IMPORTANCE_MEAN_5FOLD.csv",
            aggregated,
        )

ensemble_summary = {
    "experiment": "HOME_CREDIT_FOUR_MODEL_OPTUNA_GPU_5FOLD_IMPUTED_EXT_SOURCE_1_2_3_PLUS_9_SHARED_GMM",
    "completed_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "config_sha256": OPTUNA_MODEL_CONFIG_HASH,
    "preprocessing_config_sha256": CONFIG_HASH,
    "rows": {
        "train": n_train,
        "test": n_test,
        "positive_train": int(y.sum()),
        "positive_rate": float(y.mean()),
    },
    "cross_validation": {
        "method": "StratifiedKFold",
        "n_splits": N_SPLITS,
        "shuffle": True,
        "random_state": RANDOM_SEED,
        "same_folds_for_all_models": True,
        "verified_against_saved_imputation_folds": True,
        "optuna_inner_method": "StratifiedKFold(5) within each outer-training fold",
        "outer_validation_used_for_optuna": False,
        "outer_validation_used_for_realmlp_epoch_selection": False,
    },
    "devices": {
        "realmlp": "CUDA GPU 0",
        "catboost": "CUDA GPU 0",
        "lightgbm": "CPU",
        "xgboost": "CUDA GPU 0",
    },
    "cpu_threads": CPU_THREADS,
    "feature_policy": {
        "winning_ext_source_scenario": "IMPUTE_EXT_SOURCE_1_2_3",
        "imputed_columns": EXT_SOURCE_COLUMNS,
        "original_missing_flags": EXT_SOURCE_MISSING_COLUMNS,
        "shared_added_features": REQUESTED_SHARED_FEATURES,
        "additional_numeric_features": [
            "DIFF_RATIO", "NET_INCOME_PER_MEMBER", "CAR_FRESHNESS"
        ],
        "career_profile": {
            "feature": CAREER_PROFILE_FEATURE,
            "minimum_training_count": CAREER_PROFILE_MIN_COUNT,
            "pensioner_override": CAREER_PROFILE_PENSIONER_VALUE,
            "fallback_order": [
                "INCOME+OCCUPATION+ORGANIZATION",
                "INCOME+OCCUPATION",
                "INCOME+ORGANIZATION",
                "INCOME",
            ],
            "fit_scope": "outer training only; re-fitted in every Optuna inner split and RealMLP epoch-selection split",
        },
        "goods_price_to_annuity": {
            "classifier_feature": False,
            "imputer_input_feature": True,
        },
        "local_interaction_uses_imputed_ext_source": True,
        "local_interaction_shared_between_views": True,
        "tree_engineered_features": TREE_ENGINEERED_FEATURES,
        "realmlp_engineered_features": MLP_ENGINEERED_FEATURES,
        "career_profile_minimum_training_count": CAREER_PROFILE_MIN_COUNT,
        "career_profile_fallback_order": [
            "INCOME+OCCUPATION+ORGANIZATION",
            "INCOME+OCCUPATION",
            "INCOME+ORGANIZATION",
            "INCOME",
        ],
        "separate_tree_and_realmlp_feature_views": True,
        "gmm_fit_once_per_fold_and_shared": True,
        "gmm_persistent_cache_directory": str(GMM_CACHE_DIR),
        "neighbor_count": N_NEIGHBORS,
        "neighbor_reference_fold_train_only": True,
        "training_rows_exclude_self_from_neighbors": True,
    },
    "optuna": OPTUNA_SUMMARY,
    "optuna_configuration": OPTUNA_MODEL_CONFIG,
    "optuna_trials_per_model_per_outer_fold": OPTUNA_TRIALS_PER_MODEL_PER_FOLD,
    "optuna_inner_folds": OPTUNA_INNER_N_SPLITS,
    "realmlp_epoch_selection_audit": str(OUTPUT_DIR / "realmlp_epoch_selection_by_fold.csv"),
    "ensemble_weights": ENSEMBLE_WEIGHTS,
    "individual_pooled_oof_metrics": model_oof_metrics,
    "ensemble_pooled_oof_metrics_secondary": pooled_metrics,
    "ensemble_mean_fold_roc_auc": float(fold_auc.mean()),
    "ensemble_std_fold_roc_auc_ddof1": float(fold_auc.std(ddof=1)),
    "ensemble_pair_weighted_fold_roc_auc": pair_weighted_auc,
    "ensemble_fold_auc": fold_auc.tolist(),
    "output_directory": str(OUTPUT_DIR),
}
atomic_json(ENSEMBLE_DIR / "ensemble_summary.json", ensemble_summary)

# Final submission validation.
final_submission_path = ENSEMBLE_DIR / "submission_four_model_equal_weight.csv"
final_submission = pd.read_csv(final_submission_path)
if final_submission.shape != sample_sub.shape:
    raise RuntimeError("Final submission shape differs from sample submission")
if list(final_submission.columns) != list(sample_sub.columns):
    raise RuntimeError("Final submission columns/order differ from sample submission")
if not np.array_equal(final_submission[ID_COL].to_numpy(), sample_sub[ID_COL].to_numpy()):
    raise RuntimeError("Final submission ID order differs from sample submission")
if final_submission[TARGET_COL].isna().any() or not final_submission[TARGET_COL].between(0, 1).all():
    raise RuntimeError("Final submission contains invalid probabilities")

final_artifacts = [
    ENSEMBLE_DIR / "ensemble_summary.json",
    ENSEMBLE_DIR / "oof_predictions_four_models.csv",
    ENSEMBLE_DIR / "test_predictions_all_folds_and_models.csv",
    ENSEMBLE_DIR / "submission_four_model_equal_weight.csv",
    ENSEMBLE_DIR / "four_model_equal_weight_five_fold_metrics.csv",
    OPTUNA_DIR / "optuna_config.json",
    OPTUNA_DIR / "OPTUNA_SUMMARY.json",
    OUTPUT_DIR / "realmlp_epoch_selection_by_fold.csv",
]
mark_complete(
    CHECKPOINT_DIR / "EXPERIMENT_COMPLETE.json",
    "experiment_complete",
    OPTUNA_MODEL_CONFIG_HASH,
    final_artifacts,
)

print_section("FOUR-MODEL OPTUNA GPU/CPU ENSEMBLE COMPLETED SUCCESSFULLY")
print(f"Mean 5-fold ensemble ROC-AUC        : {fold_auc.mean():.9f}")
print(f"Std 5-fold ensemble ROC-AUC         : {fold_auc.std(ddof=1):.9f}")
print(f"Pair-weighted fold ensemble ROC-AUC : {pair_weighted_auc:.9f}")
print(f"Pooled OOF ensemble ROC-AUC         : {pooled_metrics['roc_auc']:.9f}")
print(f"Equal ensemble weights              : {ENSEMBLE_WEIGHTS}")
print(f"Optuna studies                      : {OPTUNA_STUDY_DIR}")
print(f"RealMLP epoch audit                  : {OUTPUT_DIR / 'realmlp_epoch_selection_by_fold.csv'}")
print(f"Submission                          : {final_submission_path}")
print(f"OOF predictions                     : {ENSEMBLE_DIR / 'oof_predictions_four_models.csv'}")
print(f"Fold metrics                        : {ENSEMBLE_DIR / 'four_model_equal_weight_five_fold_metrics.csv'}")
print(f"Output directory                    : {OUTPUT_DIR}")
print("RealMLP/CatBoost/XGBoost used CUDA; LightGBM used CPU. All four models were tuned with inner 5-fold Optuna CV.", flush=True)
