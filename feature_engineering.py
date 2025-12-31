"""
feature_engineering_selection.py
Feature Engineering & Feature Selection for PIL Framework

Feature Engineering:
- PatchSight CNN embeddings (InceptionResNetV2)
- Spatial Attention Pooling
- ImmuneMap Immunoscore (lymphocyte density)
- Standardized clinical covariates

Feature Selection:
- LASSO (α = 0.01)
- PCA (95% variance retention)
"""

# =====================================================
# 1. IMPORTS
# =====================================================
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Multiply,
    Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# =====================================================
# 2. STAGE-I : PATCHSIGHT FEATURE ENGINEERING
# =====================================================
def spatial_attention_pooling(feature_maps):
    """
    Spatial Attention Pooling
    """
    attention = tf.reduce_mean(feature_maps, axis=-1, keepdims=True)
    attention = tf.nn.softmax(attention, axis=(1, 2))
    weighted = feature_maps * attention
    pooled = tf.reduce_sum(weighted, axis=(1, 2))
    return pooled


def extract_patch_features(image_dir):
    """
    Extract CNN embeddings from histopathology patches
    """
    base_model = InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(299, 299, 3)
    )

    inputs = base_model.input
    x = base_model.output
    pooled = tf.keras.layers.Lambda(spatial_attention_pooling)(x)
    model = Model(inputs, pooled)

    features = []
    filenames = []

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img = load_img(img_path, target_size=(299, 299))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        embedding = model.predict(img, verbose=0)
        features.append(embedding.flatten())
        filenames.append(img_name)

    return np.array(features), filenames


# =====================================================
# 3. STAGE-II : IMMUNEMAP FEATURE ENGINEERING
# =====================================================
def compute_immunoscore(bounding_boxes, image_area):
    """
    Immunoscore = lymphocyte density
    """
    lymphocyte_count = len(bounding_boxes)
    density = lymphocyte_count / image_area
    return density


def extract_immunemap_features(detection_results):
    """
    detection_results:
    dict {image_id: list of bounding boxes}
    """
    scores = []

    for img_id, boxes in detection_results.items():
        image_area = 512 * 512
        score = compute_immunoscore(boxes, image_area)
        scores.append(score)

    return np.array(scores).reshape(-1, 1)


# =====================================================
# 4. STAGE-III : CLINICAL FEATURE ENGINEERING
# =====================================================
def preprocess_clinical_data(csv_path):
    """
    Standardizes clinical covariates
    """
    df = pd.read_csv(csv_path)

    clinical_features = df.drop(columns=["time", "event"])
    scaler = StandardScaler()
    clinical_scaled = scaler.fit_transform(clinical_features)

    return clinical_scaled, df["time"].values, df["event"].values


# =====================================================
# 5. FEATURE FUSION (PIL FRAMEWORK)
# =====================================================
def fuse_features(patch_features, immune_features, clinical_features):
    """
    Concatenates PatchSight + ImmuneMap + Clinical features
    """
    return np.concatenate(
        [patch_features, immune_features, clinical_features],
        axis=1
    )


# =====================================================
# 6. FEATURE SELECTION
# =====================================================
def lasso_feature_selection(X, y, alpha=0.01):
    """
    Sparse feature selection using LASSO
    """
    lasso = Lasso(alpha=alpha, max_iter=5000)
    lasso.fit(X, y)

    selected = np.where(lasso.coef_ != 0)[0]
    X_selected = X[:, selected]

    print(f"LASSO selected {X_selected.shape[1]} features")
    return X_selected, selected


def pca_dimensionality_reduction(X, variance=0.95):
    """
    PCA retaining specified variance
    """
    pca = PCA(n_components=variance)
    X_reduced = pca.fit_transform(X)

    print(f"PCA reduced features to {X_reduced.shape[1]} dimensions")
    return X_reduced, pca


# =====================================================
# 7. MAIN PIPELINE
# =====================================================
def main():
    print("\nStarting Feature Engineering & Selection Pipeline...\n")

    # ---- PatchSight ----
    patch_features, _ = extract_patch_features("data/breakhis_patches/malignant")
    print(f"PatchSight features shape: {patch_features.shape}")

    # ---- ImmuneMap ----
    dummy_detections = {
        f"img_{i}": [(10, 10, 30, 30)] * np.random.randint(5, 20)
        for i in range(patch_features.shape[0])
    }
    immune_features = extract_immunemap_features(dummy_detections)
    print(f"ImmuneMap features shape: {immune_features.shape}")

    # ---- Clinical ----
    clinical_features, time, event = preprocess_clinical_data(
        "data/metabric_processed.csv"
    )
    print(f"Clinical features shape: {clinical_features.shape}")

    # ---- Feature Fusion ----
    X = fuse_features(patch_features, immune_features, clinical_features)
    print(f"Fused feature shape: {X.shape}")

    # ---- Feature Selection ----
    X_lasso, selected_idx = lasso_feature_selection(X, time, alpha=0.01)
    X_final, pca_model = pca_dimensionality_reduction(X_lasso, variance=0.95)

    np.save("data/final_features.npy", X_final)
    print("\nFinal selected features saved successfully.")

    print("\nFeature Engineering & Selection completed.\n")


if __name__ == "__main__":
    main()
