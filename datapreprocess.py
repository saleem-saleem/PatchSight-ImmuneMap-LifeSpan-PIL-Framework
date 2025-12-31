"""
datapreprocess.py
Unified Data Preprocessing for PIL Framework

Stage-I  : PatchSight (Histopathology Patch Extraction)
Stage-II : ImmuneMap (IHC Image & Annotation Preparation)
Stage-III: LifeSpan (Clinical & Survival Data Processing)
"""

# =====================================================
# 1. IMPORTS
# =====================================================
import os
import cv2
import numpy as np
import pandas as pd
from patchify import patchify
from sklearn.preprocessing import StandardScaler, LabelEncoder


# =====================================================
# 2. GLOBAL SETTINGS
# =====================================================
# Stage-I
PATCH_SIZE = 300
STRIDE = 150
RESIZE_DIM = 460
PATCH_OUTPUT_DIR = "data/breakhis_patches"

# Stage-II
IHC_RESIZE = 512

# Stage-III
SURVIVAL_OUTPUT = "data/metabric_processed.csv"


# =====================================================
# 3. STAGE-I : PATCHSIGHT DATA PREPROCESSING
# =====================================================
def preprocess_histopathology(input_dir):
    """
    Extracts overlapping patches from histopathology images
    Directory structure expected:
    input_dir/
        benign/
        malignant/
    """

    print("[Stage-I] Preprocessing Histopathology Images...")

    for label in ["benign", "malignant"]:
        label_dir = os.path.join(input_dir, label)
        save_dir = os.path.join(PATCH_OUTPUT_DIR, label)
        os.makedirs(save_dir, exist_ok=True)

        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM))

            patches = patchify(
                img,
                (PATCH_SIZE, PATCH_SIZE, 3),
                step=STRIDE
            )

            count = 0
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    patch = patches[i, j, 0]
                    patch_name = f"{os.path.splitext(img_name)[0]}_{count}.png"
                    cv2.imwrite(
                        os.path.join(save_dir, patch_name),
                        cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                    )
                    count += 1

    print("[Stage-I] Patch extraction completed.")


# =====================================================
# 4. STAGE-II : IMMUNEMAP DATA PREPROCESSING
# =====================================================
def preprocess_ihc_images(image_dir, annotation_dir, output_dir):
    """
    Resizes IHC images and validates bounding box annotations
    Annotation format: xmin ymin xmax ymax (per line)
    """

    print("[Stage-II] Preprocessing IHC Images & Annotations...")
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        ann_path = os.path.join(
            annotation_dir,
            img_name.replace(".jpg", ".txt")
        )

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IHC_RESIZE, IHC_RESIZE))

        cv2.imwrite(
            os.path.join(output_dir, img_name),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )

        # Validate annotation
        if os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                for line in f:
                    xmin, ymin, xmax, ymax = map(float, line.split())
                    assert xmax > xmin and ymax > ymin, "Invalid bounding box"

    print("[Stage-II] IHC preprocessing completed.")


# =====================================================
# 5. STAGE-III : LIFESPAN DATA PREPROCESSING
# =====================================================
def preprocess_survival_data(csv_path):
    """
    Prepares survival dataset:
    - Encodes categorical variables
    - Normalizes features
    - Outputs cleaned CSV
    Required columns: time, event
    """

    print("[Stage-III] Preprocessing Survival Data...")

    df = pd.read_csv(csv_path)

    # Encode categorical columns
    for col in df.columns:
        if df[col].dtype == "object":
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))

    # Separate survival columns
    time = df["time"]
    event = df["event"]
    features = df.drop(columns=["time", "event"])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    processed_df = pd.DataFrame(features_scaled, columns=features.columns)
    processed_df["time"] = time.values
    processed_df["event"] = event.values

    processed_df.to_csv(SURVIVAL_OUTPUT, index=False)
    print(f"[Stage-III] Processed survival data saved to {SURVIVAL_OUTPUT}")


# =====================================================
# 6. MAIN EXECUTION
# =====================================================
def main():
    os.makedirs(PATCH_OUTPUT_DIR, exist_ok=True)

    # ---- Stage-I ----
    preprocess_histopathology("data/breakhis_raw")

    # ---- Stage-II ----
    preprocess_ihc_images(
        image_dir="data/lysto/raw_images",
        annotation_dir="data/lysto/raw_annotations",
        output_dir="data/lysto/images"
    )

    # ---- Stage-III ----
    preprocess_survival_data("data/metabric_raw.csv")

    print("\nAll data preprocessing stages completed successfully.")


if __name__ == "__main__":
    main()
