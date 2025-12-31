"""
visualization.py
Visualization utilities for PIL Framework

Stage-I  : Grad-CAM (PatchSight Explainability)
Stage-II : Bounding Box Visualization (ImmuneMap)
Stage-III: Kaplan–Meier Survival Curves (LifeSpan)
"""

# =====================================================
# 1. IMPORTS
# =====================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

from lifelines import KaplanMeierFitter
from torchvision.transforms import functional as F
from tensorflow.keras.models import Model


# =====================================================
# 2. STAGE-I : GRAD-CAM (PATCHSIGHT)
# =====================================================
def generate_gradcam(model, image, layer_name):
    """
    Generates Grad-CAM heatmap for a given image
    """
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-6

    return heatmap.numpy()


def overlay_gradcam(original_img, heatmap, alpha=0.4):
    """
    Overlays Grad-CAM heatmap on original image
    """
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
    return overlay


# =====================================================
# 3. STAGE-II : IMMUNEMAP BOUNDING BOX VISUALIZATION
# =====================================================
def draw_bounding_boxes(image_path, boxes, scores=None, threshold=0.5):
    """
    Draws predicted lymphocyte bounding boxes on IHC image
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i, box in enumerate(boxes):
        if scores is not None and scores[i] < threshold:
            continue

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title("ImmuneMap: Lymphocyte Detection")
    plt.show()


# =====================================================
# 4. STAGE-III : KAPLAN–MEIER SURVIVAL CURVES
# =====================================================
def plot_kaplan_meier(time, event, risk_scores):
    """
    Plots Kaplan–Meier curves for high-risk vs low-risk groups
    """
    median_risk = np.median(risk_scores)

    high_risk = risk_scores >= median_risk
    low_risk = risk_scores < median_risk

    kmf = KaplanMeierFitter()

    plt.figure(figsize=(6, 5))

    kmf.fit(time[low_risk], event[low_risk], label="Low Risk")
    kmf.plot_survival_function(ci_show=True)

    kmf.fit(time[high_risk], event[high_risk], label="High Risk")
    kmf.plot_survival_function(ci_show=True)

    plt.title("Kaplan–Meier Survival Curves")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.show()


# =====================================================
# 5. EXAMPLE USAGE
# =====================================================
if __name__ == "__main__":
    # ---- Stage-I Grad-CAM Example ----
    print("Visualization module loaded successfully.")

    # ---- Stage-II Example ----
    example_boxes = [
        [50, 50, 120, 120],
        [200, 180, 260, 240]
    ]
    example_scores = [0.9, 0.8]

    # draw_bounding_boxes("sample/ihc.jpg", example_boxes, example_scores)

    # ---- Stage-III Example ----
    time = np.array([5, 6, 7, 10, 12, 15, 18])
    event = np.array([1, 1, 0, 1, 0, 1, 0])
    risk = np.array([0.2, 0.3, 0.1, 0.8, 0.4, 0.9, 0.5])

    plot_kaplan_meier(time, event, risk)
