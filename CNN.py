"""
tiny_cnn_knit_purl.py
A small Keras CNN to classify knit vs purl from images.

Usage:
  pip install tensorflow matplotlib numpy scikit-learn pillow
  python tiny_cnn_knit_purl.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# --------- Config ---------
DATA_DIR = "data"               # expects subfolders 'knit' and 'purl'
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
SEED = 42
EPOCHS = 8
MODEL_OUT = "knit_purl_cnn.h5"
# --------------------------

# Basic checks
if not os.path.isdir(DATA_DIR):
    raise SystemExit(f"Data folder '{DATA_DIR}' not found. Create it with subfolders 'knit' and 'purl'.")

# Data generators with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.08,
    horizontal_flip=True,
    fill_mode="reflect"
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="binary",   # knit vs purl (binary)
    subset="training",
    seed=SEED
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    seed=SEED
)

# Build a tiny CNN
def build_tiny_cnn(input_shape=IMG_SIZE + (3,)):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(16, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")   # binary classification
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

model = build_tiny_cnn()
model.summary()

# Callbacks
es = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor="val_loss")

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[es, checkpoint]
)

# Evaluate on validation set
val_gen.reset()
y_true = val_gen.classes
y_pred_probs = model.predict(val_gen, verbose=1)
y_pred = (y_pred_probs.ravel() > 0.5).astype(int)

print("Classification report (validation):")
print(classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys())))

print("Confusion matrix (validation):")
print(confusion_matrix(y_true, y_pred))

# Plot training curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend(); plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend(); plt.title("Accuracy")
plt.show()

# Demo: predict a few images from validation generator
import itertools
val_gen.reset()
x_batch, y_batch = next(val_gen)
preds = model.predict(x_batch)
labels_map = {v:k for k,v in val_gen.class_indices.items()}

plt.figure(figsize=(12,6))
for i in range(min(8, len(x_batch))):
    plt.subplot(2,4,i+1)
    plt.imshow(x_batch[i])
    true_label = labels_map[int(y_batch[i])]
    pred_label = labels_map[int(preds[i][0] > 0.5)]
    plt.title(f"True: {true_label}\nPred: {pred_label}\nP={preds[i][0]:.2f}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# Save final model (already saved by checkpoint)
print(f"Best model saved to: {MODEL_OUT}")
