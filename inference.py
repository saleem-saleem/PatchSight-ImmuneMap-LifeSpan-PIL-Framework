"""
inference.py
End-to-end inference for PIL Framework
PatchSight → ImmuneMap → LifeSpan
"""

# =====================================================
# 1. IMPORTS
# =====================================================
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf

from patchify import patchify
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model


# =====================================================
# 2. GLOBAL SETTINGS
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 299
PATCH_SIZE = 300
STRIDE = 150
IHC_SIZE = 512
SCORE_THRESH = 0.5


# =====================================================
# 3. STAGE-I : PATCHSIGHT
# =====================================================
def extract_patches(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (460, 460))

    patches = patchify(image, (PATCH_SIZE, PATCH_SIZE, 3), step=STRIDE)
    return [patches[i, j, 0] for i in range(patches.shape[0])
            for j in range(patches.shape[1])]


def build_patchsight():
    base = InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)

    for units in [1024, 512, 256, 128, 64]:
        x = Dense(units, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

    output = Dense(1, activation="sigmoid")(x)
    return Model(base.input, output)


def predict_patchsight(model, histo_image):
    patches = extract_patches(histo_image)
    scores = []

    for p in patches:
        p = cv2.resize(p, (IMG_SIZE, IMG_SIZE)) / 255.0
        p = np.expand_dims(p, axis=0)
        scores.append(model.predict(p, verbose=0)[0][0])

    return float(np.mean(scores))


# =====================================================
# 4. STAGE-II : IMMUNEMAP
# =====================================================
def build_immunemap():
    backbone = resnet_fpn_backbone("resnet101", pretrained=False)
    model = FasterRCNN(backbone, num_classes=2)
    return model


def count_lymphocytes(model, ihc_image):
    model.eval()

    image = cv2.imread(ihc_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IHC_SIZE, IHC_SIZE))
    tensor = F.to_tensor(image).to(DEVICE)

    with torch.no_grad():
        pred = model([tensor])[0]

    scores = pred["scores"].cpu().numpy()
    count = int(np.sum(scores >= SCORE_THRESH))

    return count


# =====================================================
# 5. STAGE-III : LIFESPAN
# =====================================================
class DeepHit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


def load_lifespan_models():
    df = pd.read_csv("data/metabric_survival.csv")

    X = df.drop(columns=["time", "event"])
    time = df["time"].values
    event = df["event"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cox PH
    cox_df = pd.DataFrame(X_scaled)
    cox_df["time"] = time
    cox_df["event"] = event

    cox = CoxPHFitter()
    cox.fit(cox_df, "time", "event")

    # RSF
    y = Surv.from_arrays(event.astype(bool), time)
    rsf = RandomSurvivalForest(n_estimators=300, random_state=42)
    rsf.fit(X_scaled, y)

    # DeepHit
    deephit = DeepHit(X_scaled.shape[1]).to(DEVICE)
    deephit.load_state_dict(torch.load("models/deephit_lifespan.pth"))
    deephit.eval()

    return cox, rsf, deephit, scaler


# =====================================================
# 6. MAIN INFERENCE PIPELINE
# =====================================================
def run_inference(histo_image, ihc_image, clinical_features):
    print("\nRunning PIL Inference Pipeline...")

    # Load models
    patchsight = build_patchsight()
    patchsight.load_weights("models/patchsight_classifier.h5")

    immunemap = build_immunemap().to(DEVICE)
    immunemap.load_state_dict(torch.load("models/immunemap_fasterrcnn.pth"))

    cox, rsf, deephit, scaler = load_lifespan_models()

    # Stage-I
    diagnosis_prob = predict_patchsight(patchsight, histo_image)

    # Stage-II
    til_count = count_lymphocytes(immunemap, ihc_image)

    # Feature fusion
    features = np.array([[diagnosis_prob, til_count] + clinical_features])
    features_scaled = scaler.transform(features)

    # Stage-III
    cox_risk = float(cox.predict_partial_hazard(pd.DataFrame(features_scaled)).values[0])
    rsf_risk = float(rsf.predict(features_scaled)[0])
    deephit_risk = float(
        deephit(torch.tensor(features_scaled, dtype=torch.float32).to(DEVICE)).item()
    )

    return {
        "Malignancy Probability": diagnosis_prob,
        "TIL Count": til_count,
        "Cox Risk Score": cox_risk,
        "RSF Risk Score": rsf_risk,
        "DeepHit Risk Score": deephit_risk
    }


# =====================================================
# 7. EXECUTION
# =====================================================
if __name__ == "__main__":
    results = run_inference(
        histo_image="sample/histo.jpg",
        ihc_image="sample/ihc.jpg",
        clinical_features=[45, 2, 1]  # age, stage, grade
    )

    print("\nInference Results:")
    for k, v in results.items():
        print(f"{k}: {v}")
