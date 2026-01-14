from file_picker import select_image_file
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plot_pattern as grid

def predict_knit_pattern(model, class_labels, color_map):

    file_path = select_image_file()
    if not file_path:
        print("No file selected.")
    else:
        print("Selected:", file_path)
        img = Image.open(file_path).convert("RGB").resize((128, 128))
        img_array = np.array(img) / 255.0  # (128,128,3)
        img_batch = np.expand_dims(img_array, axis=0)  # (1,128,128,3)

        # Predict
        pred = model.predict(img_batch)[0]  # (128,128,num_classes)
        pred_mask = np.argmax(pred, axis=-1)

        plt.figure(figsize=(10,4))

        # Input image
        plt.subplot(1,2,1)
        plt.imshow(np.array(img))
        plt.title("Input")

        # Ground truth mask (already integer labels)
        #plt.subplot(1,3,2)
        #plt.imshow(y_val[0], cmap='tab20')   # <-- no argmax here
        #plt.title("True Mask")

        # Predicted mask
        plt.subplot(1,2,2)
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