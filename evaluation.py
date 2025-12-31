"""
metrics.py
Unified evaluation metrics for PIL Framework

Stage-I  : PatchSight (Classification Metrics)
Stage-II : ImmuneMap (Detection Metrics)
Stage-III: LifeSpan (Survival Metrics)
"""

# =====================================================
# 1. IMPORTS
# =====================================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc


# =====================================================
# 2. STAGE-I : PATCHSIGHT METRICS
# =====================================================
def classification_metrics(y_true, y_pred, y_prob=None):
    """
    Computes standard classification metrics
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred)
    }

    if y_prob is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)

    return metrics


def plot_confusion_matrix(y_true, y_pred, labels=("Benign", "Malignant")):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.show()


# =====================================================
# 3. STAGE-II : IMMUNEMAP METRICS
# =====================================================
def iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def detection_metrics(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Computes Precision, Recall, F1 for object detection
    """
    tp = 0
    fp = 0
    fn = len(gt_boxes)

    for pbox in pred_boxes:
        matched = False
        for gt in gt_boxes:
            if iou(pbox, gt) >= iou_threshold:
                matched = True
                break

        if matched:
            tp += 1
            fn -= 1
        else:
            fp += 1

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }


# =====================================================
# 4. STAGE-III : LIFESPAN METRICS
# =====================================================
def survival_metrics(time, event, risk_scores):
    """
    Computes Concordance Index (C-index)
    """
    c_index = concordance_index(time, -risk_scores, event)

    return {
        "C-Index": c_index
    }


def time_dependent_auc(train_time, train_event,
                       test_time, test_event,
                       test_risk_scores, times):
    """
    Computes Time-dependent AUC
    """
    auc, mean_auc = cumulative_dynamic_auc(
        np.array(list(zip(train_event, train_time)),
                 dtype=[("event", bool), ("time", float)]),
        np.array(list(zip(test_event, test_time)),
                 dtype=[("event", bool), ("time", float)]),
        test_risk_scores,
        times
    )

    return auc, mean_auc


# =====================================================
# 5. EXAMPLE USAGE
# =====================================================
if __name__ == "__main__":
    # ---- Stage-I Example ----
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    y_prob = [0.2, 0.9, 0.4, 0.1, 0.8]

    cls_metrics = classification_metrics(y_true, y_pred, y_prob)
    print("\nPatchSight Metrics:", cls_metrics)
    plot_confusion_matrix(y_true, y_pred)

    # ---- Stage-II Example ----
    gt_boxes = [[10, 10, 50, 50], [100, 100, 140, 140]]
    pred_boxes = [[12, 12, 48, 48], [200, 200, 240, 240]]

    det_metrics = detection_metrics(gt_boxes, pred_boxes)
    print("\nImmuneMap Metrics:", det_metrics)

    # ---- Stage-III Example ----
    time = np.array([5, 6, 7, 10, 12])
    event = np.array([1, 1, 0, 1, 0])
    risk = np.array([0.8, 0.7, 0.2, 0.9, 0.3])

    surv_metrics = survival_metrics(time, event, risk)
    print("\nLifeSpan Metrics:", surv_metrics)
