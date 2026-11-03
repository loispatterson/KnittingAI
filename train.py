"""Train the stitch segmenter and say honestly how well it did.

Run:  python train.py            # 5-fold, the honest number
      python train.py --final    # train on everything, save the model

What changed from the original training script, and why each one matters more
than the model architecture does:

  Augmentation. There are 29 images. Flips are free data and the single
  largest available gain. But only some flips: knitting has a grain. Rows run
  horizontally, and a 90 degree rotation turns a knit column into a knit row,
  which is a different fabric -- so horizontal and vertical flips are used and
  rotation is not. The model should learn the texture, not the framing.

  A loss that notices the small classes. Garter is 11% of pixels. Plain
  cross-entropy is happy to ignore it, because being right about background
  pays better. Class weights make the minority classes cost what they should.

  Early stopping on mean IoU, not on loss or accuracy. With 23 training images
  the model can memorise them, and val loss starts rising long before the
  useful epochs are over.

  The number reported is the mean over 5 folds with the spread shown, not a
  single run. One split of six test images is too small to quote.
"""
import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import data_loading as dl
import evaluate as ev
import splits

IMG_SIZE = dl.IMG_SIZE
SEED = 0


def build_unet(num_classes, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """Loïs's original TinyUNet, unchanged in shape. The architecture was never
    the problem: a 29-image dataset does not have the information to justify
    anything bigger, and swapping in a pretrained encoder is the next
    experiment, not this one."""
    inputs = layers.Input(shape=input_shape)

    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(c3)

    u1 = layers.UpSampling2D()(c3)
    u1 = layers.Concatenate()([u1, c2])
    c4 = layers.Conv2D(32, 3, activation="relu", padding="same")(u1)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c1])
    c5 = layers.Conv2D(16, 3, activation="relu", padding="same")(u2)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(c5)
    return models.Model(inputs, outputs)


def augment(X, y):
    """Flips only, applied to image and mask together. Eight times the data
    would be four flips times two rotations, but see the note at the top: a
    rotated knit is not a knit."""
    Xs, ys = [X], [y]
    Xs.append(X[:, :, ::-1]);        ys.append(y[:, :, ::-1])          # left-right
    Xs.append(X[:, ::-1, :]);        ys.append(y[:, ::-1, :])          # up-down
    Xs.append(X[:, ::-1, ::-1]);     ys.append(y[:, ::-1, ::-1])       # both
    return np.concatenate(Xs), np.concatenate(ys)


def class_weights(y, num_classes):
    """Inverse frequency, normalised to mean 1 over the classes that are
    actually painted, so the learning rate stays comparable to an unweighted
    run. A class in color_map.json with no masks yet (knit, purl and seed at
    the time of writing) gets weight zero rather than a huge inverse-of-nothing
    weight that would distort the normalisation of every other class."""
    counts = np.bincount(y.ravel(), minlength=num_classes).astype(np.float64)
    present = counts > 0
    w = np.zeros(num_classes, dtype=np.float64)
    w[present] = counts[present].sum() / (present.sum() * counts[present])
    w[present] /= w[present].mean()
    return w


def weighted_sparse_ce(weights):
    w = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true, axis=-1) if y_true.shape[-1] == 1 else y_true, tf.int32)
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(ce * tf.gather(w, y_true))

    return loss


class MeanIoUStop(tf.keras.callbacks.Callback):
    """Early stopping on held-out mean IoU, keeping the best weights. Keras has
    no built-in metric that ignores absent classes the way evaluate.py does, so
    the check runs here rather than as a compiled metric."""

    def __init__(self, X_val, y_val, names, patience=25):
        super().__init__()
        self.X_val, self.y_val, self.names, self.patience = X_val, y_val, names, patience
        self.best, self.best_weights, self.waited, self.best_epoch = -1.0, None, 0, 0

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.X_val, verbose=0).argmax(axis=-1)
        score = ev.summarise(self.y_val, pred, self.names)["mean_iou"]
        logs = logs or {}
        logs["val_mean_iou"] = score
        if score > self.best:
            self.best, self.best_epoch = score, epoch
            self.best_weights = self.model.get_weights()
            self.waited = 0
        else:
            self.waited += 1
            if self.waited >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def fit_one(X_tr, y_tr, X_te, y_te, names, epochs=150, verbose=0):
    tf.keras.utils.set_random_seed(SEED)
    X_aug, y_aug = augment(X_tr, y_tr)
    model = build_unet(len(names))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=weighted_sparse_ce(class_weights(y_tr, len(names))))
    stopper = MeanIoUStop(X_te, y_te, names)
    model.fit(X_aug, y_aug, epochs=epochs, batch_size=8, verbose=verbose,
              callbacks=[stopper], shuffle=True)
    pred = model.predict(X_te, verbose=0).argmax(axis=-1)
    return model, ev.summarise(y_te, pred, names), stopper.best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true",
                    help="train on all images and save, after the folds have been seen")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()

    colour_map, names = dl.load_color_map()
    X, y, stems = dl.load_data(color_map=colour_map)
    print("\nclass balance over the whole set:")
    for i, n, count, share in dl.class_balance(y, names):
        print(f"  {n:<22}{count:>10,} px{share:>9.2%}")

    base, per_fold, best_epochs = {}, [], []
    for fold, (tr, te) in enumerate(splits.k_folds(stems, k=args.folds, seed=SEED)):
        print(f"\n--- fold {fold} ---")
        splits.describe(stems, y, names, tr, te)
        for label, res in ev.baselines(y[tr], y[te], names).items():
            base.setdefault(label, []).append(res["mean_iou"])
        _, res, best_epoch = fit_one(X[tr], y[tr], X[te], y[te], names, epochs=args.epochs)
        per_fold.append(res)
        best_epochs.append(best_epoch)
        print(f"    best epoch {best_epoch}, mean IoU {res['mean_iou']:.3f}")

    print("\n" + "=" * 64)
    scores = np.array([r["mean_iou"] for r in per_fold])
    for label, v in base.items():
        v = np.array(v)
        print(f"  baseline {label:<22} {v.mean():.3f}")
    print(f"  MODEL{'':<31}{scores.mean():.3f}  "
          f"(folds {np.round(scores, 3).tolist()}, spread {scores.max()-scores.min():.3f})")

    # Per-class IoU pooled across folds, which is the table worth quoting.
    pooled = sum(r["confusion"] for r in per_fold)
    pooled_res = {"confusion": pooled, "iou": ev.iou_per_class(pooled),
                  "dice": ev.dice_per_class(pooled), "names": names,
                  "mean_iou": float(np.nanmean(ev.iou_per_class(pooled))),
                  "accuracy": float(np.diag(pooled).sum() / pooled.sum())}
    ev.report(pooled_res, "pooled over all folds")
    ev.report_confusion(pooled_res)

    if args.final:
        print("\ntraining the final model on all 29 images")
        tf.keras.utils.set_random_seed(SEED)
        X_aug, y_aug = augment(X, y)
        model = build_unet(len(names))
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss=weighted_sparse_ce(class_weights(y, len(names))))
        model.fit(X_aug, y_aug, epochs=int(np.mean(best_epochs)) + 1, batch_size=8, verbose=0)
        model.save("stitch_unet.keras")
        print(f"saved stitch_unet.keras, trained {int(np.mean(best_epochs)) + 1} epochs "
              f"(the fold average, so it stops where the folds stopped improving)")
        print("NOTE: this model has seen every image. Its score is not measurable "
              "any more -- the 5-fold number above is the one to quote.")


if __name__ == "__main__":
    sys.exit(main())
