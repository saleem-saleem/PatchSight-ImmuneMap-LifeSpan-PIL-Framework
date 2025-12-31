"""
cv_and_reproducibility.py
Cross-validation, random seed control, and environment configuration
for the PIL Framework (PatchSight + ImmuneMap + LifeSpan)

Specifications:
- Stratified 5-Fold Cross-Validation (20% per fold)
- Fixed Random Seed = 42 (NumPy, TensorFlow, PyTorch, scikit-learn)
- Environment: Python 3.10, TF 2.12, PyTorch 2.1
"""

# =====================================================
# 1. IMPORTS
# =====================================================
import os
import random
import platform
import numpy as np
import pandas as pd

import tensorflow as tf
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# =====================================================
# 2. GLOBAL RANDOM SEED SETTINGS
# =====================================================
SEED = 42

def set_global_seed(seed=42):
    """
    Fixes random seeds for full reproducibility
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)
    np.random.seed(seed)

    tf.random.set_seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Global random seed set to {seed}")


# =====================================================
# 3. ENVIRONMENT INFORMATION LOGGER
# =====================================================
def log_environment():
    """
    Logs software & hardware environment
    """
    print("\n===== IMPLEMENTATION ENVIRONMENT =====")
    print(f"Python Version     : {platform.python_version()}")
    print(f"TensorFlow Version : {tf.__version__}")
    print(f"PyTorch Version    : {torch.__version__}")
    print(f"NumPy Version      : {np.__version__}")

    try:
        import sklearn
        print(f"Scikit-learn       : {sklearn.__version__}")
    except:
        pass

    try:
        import sksurv
        print(f"Scikit-survival    : {sksurv.__version__}")
    except:
        pass

    try:
        import lifelines
        print(f"Lifelines          : {lifelines.__version__}")
    except:
        pass

    print(f"CUDA Available     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device         : {torch.cuda.get_device_name(0)}")

    print("RAM                : 64 GB (Configured)")
    print("=====================================\n")


# =====================================================
# 4. STRATIFIED 5-FOLD CROSS-VALIDATION
# =====================================================
def stratified_5fold_cv(X, y):
    """
    Stratified 5-fold cross-validation
    Each fold uses 20% of data for validation
    """
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    folds = []
    fold_id = 1

    for train_idx, val_idx in skf.split(X, y):
        print(f"[Fold {fold_id}]")
        print(f"Train size: {len(train_idx)} | Validation size: {len(val_idx)}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        folds.append((X_train, X_val, y_train, y_val))
        fold_id += 1

    return folds


# =====================================================
# 5. DATA NORMALIZATION PER FOLD
# =====================================================
def scale_per_fold(X_train, X_val):
    """
    Standardizes features per fold (no data leakage)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled


# =====================================================
# 6. EXAMPLE USAGE (ALL STAGES COMPATIBLE)
# =====================================================
def main():
    print("\nInitializing Reproducible Cross-Validation Pipeline...\n")

    # ---- Seed Fixing ----
    set_global_seed(SEED)

    # ---- Environment Logging ----
    log_environment()

    # ---- Dummy Dataset (Replace with PIL features) ----
    X = np.random.rand(500, 128)      # Feature matrix
    y = np.random.randint(0, 2, 500)  # Stratification labels

    print("Dataset Shape:", X.shape)

    # ---- Stratified 5-Fold CV ----
    folds = stratified_5fold_cv(X, y)

    # ---- Per-Fold Scaling ----
    for i, (X_tr, X_va, y_tr, y_va) in enumerate(folds):
        X_tr, X_va = scale_per_fold(X_tr, X_va)
        print(f"Fold {i+1} scaled successfully")

    print("\nCross-validation setup completed successfully.")


# =====================================================
# 7. ENTRY POINT
# =====================================================
if __name__ == "__main__":
    main()
