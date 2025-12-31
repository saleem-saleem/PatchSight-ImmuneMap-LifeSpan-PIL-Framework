"""
train_all.py
Unified training script for:
Stage-I  : PatchSight (Histopathology Classification)
Stage-II : ImmuneMap (TIL Detection)
Stage-III: LifeSpan (Survival Prediction)
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
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# =====================================================
# 2. GLOBAL SETTINGS
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stage-I
IMG_SIZE = 299
BATCH_SIZE = 32
EPOCHS_STAGE1 = 30

# Stage-II
IMG_SIZE_IHC = 512
BATCH_SIZE_STAGE2 = 2
EPOCHS_STAGE2 = 20

# Stage-III
EPOCHS_STAGE3 = 100
LR_STAGE3 = 1e-3


# =====================================================
# 3. STAGE-I : PATCHSIGHT TRAINING
# =====================================================
def train_patchsight():
    print("\n[Stage-I] Training PatchSight Classifier...")

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.1,
        validation_split=0.1
    )

    train_gen = datagen.flow_from_directory(
        "data/breakhis_patches",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        "data/breakhis_patches",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation"
    )

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
    model = Model(base.input, output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_STAGE1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3)
        ]
    )

    model.save("models/patchsight_classifier.h5")
    print("[Stage-I] PatchSight model saved.")


# =====================================================
# 4. STAGE-II : IMMUNEMAP TRAINING
# =====================================================
class ImmunemapDataset(Dataset):
    def __init__(self, img_dir, ann_dir):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE_IHC, IMG_SIZE_IHC))
        img = F.to_tensor(img)

        boxes = []
        labels = []
        ann_path = os.path.join(self.ann_dir, img_name.replace(".jpg", ".txt"))

        if os.path.exists(ann_path):
            with open(ann_path) as f:
                for line in f:
                    xmin, ymin, xmax, ymax = map(float, line.split())
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def train_immunemap():
    print("\n[Stage-II] Training ImmuneMap (Faster R-CNN)...")

    dataset = ImmunemapDataset(
        "data/lysto/images",
        "data/lysto/annotations"
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_STAGE2,
        shuffle=True,
        collate_fn=collate_fn
    )

    backbone = resnet_fpn_backbone("resnet101", pretrained=True)
    model = FasterRCNN(backbone, num_classes=2)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(EPOCHS_STAGE2):
        total_loss = 0
        for imgs, targets in loader:
            imgs = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS_STAGE2}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "models/immunemap_fasterrcnn.pth")
    print("[Stage-II] ImmuneMap model saved.")


# =====================================================
# 5. STAGE-III : LIFESPAN TRAINING
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


def train_lifespan():
    print("\n[Stage-III] Training LifeSpan Survival Models...")

    df = pd.read_csv("data/metabric_survival.csv")

    X = df.drop(columns=["time", "event"])
    time = df["time"].values
    event = df["event"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cox PH
    cox = CoxPHFitter()
    df_cox = pd.DataFrame(X_scaled)
    df_cox["time"] = time
    df_cox["event"] = event
    cox.fit(df_cox, "time", "event")

    # RSF
    y = Surv.from_arrays(event.astype(bool), time)
    rsf = RandomSurvivalForest(n_estimators=300, random_state=42)
    rsf.fit(X_scaled, y)

    # DeepHit
    model = DeepHit(X_scaled.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_STAGE3)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    t_t = torch.tensor(time, dtype=torch.float32).to(DEVICE)

    for epoch in range(EPOCHS_STAGE3):
        pred = model(X_t).squeeze()
        loss = loss_fn(pred, t_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS_STAGE3}] Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "models/deephit_lifespan.pth")
    print("[Stage-III] LifeSpan models trained and saved.")


# =====================================================
# 6. MAIN EXECUTION
# =====================================================
def main():
    os.makedirs("models", exist_ok=True)

    train_patchsight()
    train_immunemap()
    train_lifespan()

    print("\nAll stages trained successfully!")


if __name__ == "__main__":
    main()
