import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2
import glob
import matplotlib.pyplot as plt
import plot_pattern as grid
from matplotlib.colors import to_hex
import json

IMG_SIZE = 128
NUM_CLASSES = 4  # background, stockinette, reverse_stockinette, garter

# Load images and masks
# def load_data(img_dir, mask_dir):
#     images, masks = [], []
#     for img_path in glob.glob(os.path.join(img_dir, "*.jpg")):
#         name = os.path.basename(img_path).replace(".jpg", "")
#         mask_path = os.path.join(mask_dir, f"{name}_mask.png")
#         if not os.path.exists(mask_path):
#             continue
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#         img = img / 255.0
#         #converts to 8bit grey scale
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
#         # Convert RGB to a single integer label map
#         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
#         unique_colors = np.unique(mask.reshape(-1, 3), axis=0)
#
#         # Build a mapping dict automatically
#         color_to_label = {tuple(c): i for i, c in enumerate(unique_colors)}
#         print(color_to_label)
#         mask_labels = np.zeros(mask.shape[:2], dtype=np.uint8)
#
#         for color, idx in color_to_label.items():
#             mask_labels[(mask == color).all(axis=-1)] = idx
#             #print("Colour: " + str(color) + "-> index: " + str(idx) )
#         mask = mask_labels
#
#         images.append(img)
#         masks.append(mask)
#     X = np.array(images)
#     y = np.array(masks)
#
#     y = tf.keras.utils.to_categorical(y, NUM_CLASSES)
#     return X, y
#


def load_data(image_dir, mask_dir, color_map=None, class_labels=None, img_size=128):
    """
    Loads images and segmentation masks, converts colors to class labels,
    and prints the color-to-class mapping used.

    Parameters
    ----------
    image_dir : str
        Directory with input images.
    mask_dir : str
        Directory with colored segmentation masks.
    color_map : dict or None
        Mapping of RGB tuples to class IDs. If None, it will auto-detect.
    img_size : int
        Resize all images and masks to this square size.
    """

    X, y = [], []
    all_colors = set()

    # Get sorted file lists
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))])
    mask_files  = sorted([f for f in os.listdir(mask_dir) if f.endswith((".jpg", ".png"))])

    if len(image_files) != len(mask_files):
        print("⚠️  Warning: number of images and masks differ!")

    # Load each image/mask pair
    for img_name, mask_name in zip(image_files, mask_files):
        img_path  = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)

        img  = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        img  = cv2.resize(img, (img_size, img_size))
        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        # Convert to RGB
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Collect unique colors in this mask
       # unique_colors = np.unique(mask.reshape(-1, 3), axis=0)
       # for c in unique_colors:
        #    all_colors.add(tuple(c))

        # If no mapping given, auto-assign indices in order of discovery
        #if color_map is None:
         #   color_map = {tuple(c): i for i, c in enumerate(sorted(all_colors))}

        # Create label mask
        label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        for color, idx in color_map.items():
            label_mask[(mask == color).all(axis=-1)] = idx

        X.append(img)
        y.append(label_mask)

    # Print out the mapping table
    print("\n🧶 Color mapping used:")
    for color, idx in color_map.items():
        print(f"  Class {idx}: RGB {color}  →  hex {to_hex(np.array(color)/255)}")

    #with open("color_map.json", "w") as f:
     #   json.dump({str(k): v for k, v in color_map.items()}, f, indent=2)

    X = np.array(X)
    y = np.array(y)

    return X, y

def load_color_map(json_path):
    """
    Reads a JSON color map of the form:
    {
      "class_name": {"(r,g,b)": class_id}, ...
    }
    and converts it into:
      color_map = {(r,g,b): class_id}
      class_labels = ["background", "stockinette", ...]
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    color_map = {}
    class_labels = []

    for class_name, color_dict in data.items():
        # Each color_dict has one item like {"(0,0,0)": 0}
        for color_str, idx in color_dict.items():
            # Convert "(0,0,0)" → (0, 0, 0)
            color_tuple = tuple(map(int, color_str.strip("() ").split(",")))
            color_map[color_tuple] = idx
            class_labels.append(class_name)

    return color_map, class_labels


color_map, class_labels = load_color_map("color_map.json")

X, y = load_data("data/images", "data/masks", color_map, class_labels)

# Split
split = int(len(X)*0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Define U-Net
def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(c3)

    # Decoder
    u1 = layers.UpSampling2D()(c3)
    u1 = layers.Concatenate()([u1, c2])
    c4 = layers.Conv2D(32, 3, activation="relu", padding="same")(u1)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c1])
    c5 = layers.Conv2D(16, 3, activation="relu", padding="same")(u2)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(c5)

    model = models.Model(inputs, outputs)
#    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

model = build_unet()
model.summary()

# Train
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=4)

# Predict example
# Predict one example
# Predict one example
pred = model.predict(X_val[:1])[0]  # shape (128,128,num_classes)
pred_mask = np.argmax(pred, axis=-1)  # shape (128,128)

plt.figure(figsize=(10,4))

# Input image
plt.subplot(1,3,1)
plt.imshow(X_val[0].astype(np.uint8))
plt.title("Input")

# Ground truth mask (already integer labels)
plt.subplot(1,3,2)
plt.imshow(y_val[0], cmap='tab20')   # <-- no argmax here
plt.title("True Mask")

# Predicted mask
plt.subplot(1,3,3)
plt.imshow(pred_mask, cmap='tab20')
plt.title("Predicted Mask")

plt.tight_layout()
plt.show()

# Symbol printout
symbol_map = {0: " ", 1: "V", 2: "o", 3: "u", 4: "k", 5: "p", 6: "s"}  # example mapping
for row in pred_mask:
    print("".join(symbol_map.get(int(v), "?") for v in row))

colour_map_255 = list(color_map.keys())
colour_map_0_to_1 = list(np.divide(colour_map_255,255))
grid.plot_knit_grid(
    pred_mask,
    class_colors=colour_map_0_to_1,
    class_labels=class_labels,
    title="Knit Pattern Segmentation"
)