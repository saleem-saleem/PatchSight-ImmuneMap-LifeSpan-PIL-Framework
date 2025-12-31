import os
import cv2
import numpy as np
import tensorflow as tf
from patchify import patchify
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ===============================
# 2. GLOBAL PARAMETERS
# ===============================
PATCH_SIZE = 300
STRIDE = 150
IMG_SIZE = 299
BATCH_SIZE = 32
EPOCHS = 30
DATASET_DIR = "data/breakhis_patches"


# ===============================
# 3. PATCH EXTRACTION
# ===============================
def extract_patches(image_path):
    """
    Extract overlapping patches from a histopathology image
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (460, 460))

    patches = patchify(
        image,
        (PATCH_SIZE, PATCH_SIZE, 3),
        step=STRIDE
    )

    patch_list = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j, 0]
            patch_list.append(patch)

    return patch_list


# ===============================
# 4. DATA GENERATOR
# ===============================
def get_data_generators():
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        validation_split=0.1
    )

    train_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False
    )

    return train_gen, val_gen


# ===============================
# 5. PATCHSIGHT MODEL
# ===============================
def build_patchsight_classifier():
    base_model = InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze backbone
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    for units in [1024, 512, 256, 128, 64]:
        x = Dense(
            units,
            activation="relu",
            kernel_regularizer=l2(1e-4)
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model


# ===============================
# 6. TRAINING FUNCTION
# ===============================
def train_model(model, train_gen, val_gen):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=3
        )
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    return history


# ===============================
# 7. OPTIONAL FINE-TUNING
# ===============================
def fine_tune_model(model, train_gen, val_gen):
    for layer in model.layers[-80:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10
    )


# ===============================
# 8. EVALUATION
# ===============================
def evaluate_model(model, val_gen):
    y_true = val_gen.classes
    y_pred = model.predict(val_gen)
    y_pred = (y_pred > 0.5).astype(int)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.show()


# ===============================
# 9. MAIN EXECUTION
# ===============================
def main():
    print("Initializing PatchSight Classifier...")

    train_gen, val_gen = get_data_generators()

    model = build_patchsight_classifier()
    model.summary()

    print("\nTraining Phase-I (Transfer Learning)...")
    train_model(model, train_gen, val_gen)

    print("\nFine-Tuning Phase...")
    fine_tune_model(model, train_gen, val_gen)

    print("\nEvaluating Model...")
    evaluate_model(model, val_gen)

    model.save("patchsight_classifier.h5")
    print("\nModel saved as patchsight_classifier.h5")


if __name__ == "__main__":
    main()
