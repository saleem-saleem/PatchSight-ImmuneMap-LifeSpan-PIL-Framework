ImmuneMap Stage-II
Tumor-Infiltrating Lymphocyte (TIL) Detection using Faster R-CNN
Dataset: LYSTO / IHC (CD3 / CD8)
Framework: PyTorch
"""

# ===============================
# 1. IMPORTS
# ===============================
import os
import torch
import torchvision
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import box_convert


# ===============================
# 2. GLOBAL PARAMETERS
# ===============================
IMAGE_SIZE = 512
BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PATH = "data/lysto"
NUM_CLASSES = 2  # background + lymphocyte


# ===============================
# 3. DATASET CLASS
# ===============================
class ImmunemapDataset(Dataset):
    """
    Custom Dataset for IHC lymphocyte detection
    Annotation format: xmin, ymin, xmax, ymax (txt files)
    """

    def __init__(self, image_dir, annotation_dir):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        ann_path = os.path.join(
            self.annotation_dir,
            img_name.replace(".jpg", ".txt")
        )

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = F.to_tensor(image)

        boxes = []
        labels = []

        # Load bounding boxes
        if os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                for line in f.readlines():
                    xmin, ymin, xmax, ymax = map(float, line.strip().split())
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(1)  # lymphocyte

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return image, target


# ===============================
# 4. COLLATE FUNCTION
# ===============================
def collate_fn(batch):
    return tuple(zip(*batch))


# ===============================
# 5. FASTER R-CNN MODEL
# ===============================
def build_immunemap_model():
    backbone = resnet_fpn_backbone(
        backbone_name="resnet101",
        pretrained=True
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=NUM_CLASSES
    )

    return model


# ===============================
# 6. TRAINING FUNCTION
# ===============================
def train_model(model, dataloader):
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for images, targets in dataloader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f}")


# ===============================
# 7. LYMPHOCYTE COUNT INFERENCE
# ===============================
def count_lymphocytes(model, image_path, score_thresh=0.5):
    model.eval()

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    tensor = F.to_tensor(image).to(DEVICE)

    with torch.no_grad():
        prediction = model([tensor])[0]

    scores = prediction["scores"].cpu().numpy()
    boxes = prediction["boxes"].cpu().numpy()

    valid_boxes = boxes[scores >= score_thresh]
    lymphocyte_count = len(valid_boxes)

    return lymphocyte_count, valid_boxes


# ===============================
# 8. MAIN EXECUTION
# ===============================
def main():
    print("Initializing ImmuneMap Stage-II...")

    image_dir = os.path.join(DATASET_PATH, "images")
    ann_dir = os.path.join(DATASET_PATH, "annotations")

    dataset = ImmunemapDataset(image_dir, ann_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = build_immunemap_model()
    model.to(DEVICE)

    print("\nTraining Faster R-CNN for Lymphocyte Detection...")
    train_model(model, dataloader)

    torch.save(model.state_dict(), "immunemap_fasterrcnn.pth")
    print("\nModel saved as immunemap_fasterrcnn.pth")

    # Sample inference
    sample_image = os.path.join(image_dir, dataset.images[0])
    count, boxes = count_lymphocytes(model, sample_image)

    print(f"\nDetected Lymphocytes: {count}")


if __name__ == "__main__":
    main()
