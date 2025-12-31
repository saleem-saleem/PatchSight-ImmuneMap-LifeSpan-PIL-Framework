"""
PIL MASTER PIPELINE
PatchSight (Diagnosis) → ImmuneMap (TIL Detection) → LifeSpan (Survival Prediction)
"""

# ======================================================
# 1. IMPORTS
# ======================================================
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf

from patchify import patchify
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model


# ======================================================
# 2. GLOBAL SETTINGS
# ======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 299
PATCH_SIZE = 300
STRIDE = 150


# ======================================================
# 3. STAGE-I : PATCHSIGHT (DIAGNOSIS)
# ======================================================
def extract_patches(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (460, 460))

    patches = patchify(img, (PATCH_SIZE, PATCH_SIZE, 3), step=STRIDE)
    return [patches[i, j, 0] for i in range(patches.shape[0]) for j in range(patches.shape[1])]


def build_patchsight():
    base = InceptionResNetV2(include_top=False, weights="imagenet",
                             input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)

    for u in [1024, 512, 256, 128, 64]:
        x = Dense(u, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

    output = Dense(1, activation="sigmoid")(x)
    return Model(base.input, output)


def predict_patchsight(model, image_path):
    patches = extract_patches(image_path)
    preds = []

    for p in patches:
        p = cv2.resize(p, (IMG_SIZE, IMG_SIZE)) / 255.0
        preds.append(model.predict(np.expand_dims(p, axis=0))[0][0])

    return np.mean(preds)  # malignancy probability


# ======================================================
# 4. STAGE-II : IMMUNEMAP (TIL DETECTION)
# ======================================================
def build_immunemap():
    backbone = resnet_fpn_backbone("resnet101", pretrained=True)
    model = FasterRCNN(backbone, num_classes=2)
    return model


def count_lymphocytes(model, ihc_image_path, score_thresh=0.5):
    model.eval()

    img = cv2.imread(ihc_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    tensor = F.to_tensor(img).to(DEVICE)

    with torch.no_grad():
        pred = model([tensor])[0]

    scores = pred["scores"].cpu().numpy()
    count = np.sum(scores >= score_thresh)

    return count


# ======================================================
# 5. STAGE-III : LIFESPAN (SURVIVAL)
# ======================================================
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


def train_lifespan_models(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop(columns=["time", "event"]))
    time = df["time"].values
    event = df["event"].values

    # Cox PH
    cox_df = df.copy()
    cox = CoxPHFitter()
    cox.fit(cox_df, duration_col="time", event_col="event")

    # RSF
    y = Surv.from_arrays(event.astype(bool), time)
    rsf = RandomSurvivalForest(n_estimators=300, random_state=42)
    rsf.fit(X, y)

    # DeepHit
    model = DeepHit(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    t_t = torch.tensor(time, dtype=torch.float32).to(DEVICE)

    for _ in range(100):
        pred = model(X_t).squeeze()
        loss = loss_fn(pred, t_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return cox, rsf, model, scaler


# ======================================================
# 6. MASTER PIPELINE
# ======================================================
def PIL_pipeline(histo_image, ihc_image, clinical_features, survival_models):
    patchsight_model, immunemap_model, cox, rsf, deephit, scaler = survival_models

    diagnosis_score = predict_patchsight(patchsight_model, histo_image)
    til_count = count_lymphocytes(immunemap_model, ihc_image)

    features = np.array([[diagnosis_score, til_count] + clinical_features])
    features_scaled = scaler.transform(features)

    cox_risk = cox.predict_partial_hazard(pd.DataFrame(features_scaled))
    rsf_risk = rsf.predict(features_scaled)
    deephit_risk = deephit(torch.tensor(features_scaled,
                                        dtype=torch.float32).to(DEVICE)).item()

    return {
        "Diagnosis Probability": diagnosis_score,
        "TIL Count": til_count,
        "Cox Risk": float(cox_risk.values[0]),
        "RSF Risk": float(rsf_risk[0]),
        "DeepHit Risk": deephit_risk
    }


# ======================================================
# 7. MAIN
# ======================================================
def main():
    print("Initializing PIL Master Pipeline...")

    patchsight = build_patchsight()
    immunemap = build_immunemap().to(DEVICE)

    survival_df = pd.read_csv("data/metabric_survival.csv")
    cox, rsf, deephit, scaler = train_lifespan_models(survival_df)

    survival_models = (
        patchsight,
        immunemap,
        cox,
        rsf,
        deephit,
        scaler
    )

    results = PIL_pipeline(
        histo_image="sample/histo.jpg",
        ihc_image="sample/ihc.jpg",
        clinical_features=[45, 2, 1],  # age, stage, grade
        survival_models=survival_models
    )

    print("\nPIL Prediction Output:")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
