"""How good is the model, honestly.

The original script reported `accuracy` on a 128x128 segmentation where 58% of
pixels are background. Predicting "background" for every pixel of every image
scores 58% and learns nothing, so accuracy cannot tell a working model from a
broken one. Everything here exists to make that failure impossible to miss.

Three ideas, in order of how much they matter:

  Intersection over Union, per class. For one class, IoU is the overlap between
  what the model said and what the mask says, divided by their union. A class
  the model never predicts scores 0 however common background is, so a
  collapsed model is obvious instead of flattering.

  Baselines. A score means nothing on its own. "All background" and "the most
  common class everywhere" are free, so any real model has to beat them before
  the number is worth discussing.

  The confusion matrix. Mean IoU says how wrong; the matrix says wrong in which
  direction. Knit and purl are the two faces of the same fabric and look alike,
  so confusing those two is a different problem from confusing either with
  background, and only the matrix separates them.
"""
import numpy as np


def confusion(y_true, y_pred, n_classes):
    """Rows are truth, columns are prediction, counted in pixels."""
    t = np.asarray(y_true).ravel()
    p = np.asarray(y_pred).ravel()
    keep = (t >= 0) & (t < n_classes)
    return np.bincount(t[keep] * n_classes + p[keep],
                       minlength=n_classes ** 2).reshape(n_classes, n_classes)


def iou_per_class(cm):
    """NaN, not zero, for a class with no pixels in the truth -- a class that is
    absent was not got wrong, and averaging a zero in would quietly punish the
    model for the contents of the test split."""
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(axis=1) + cm.sum(axis=0) - tp
    present = cm.sum(axis=1) > 0
    out = np.full(len(tp), np.nan)
    np.divide(tp, union, out=out, where=(union > 0) & present)
    return out


def dice_per_class(cm):
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(axis=1) + cm.sum(axis=0)
    present = cm.sum(axis=1) > 0
    out = np.full(len(tp), np.nan)
    np.divide(2 * tp, denom, out=out, where=(denom > 0) & present)
    return out


def summarise(y_true, y_pred, names):
    cm = confusion(y_true, y_pred, len(names))
    iou = iou_per_class(cm)
    return {
        "confusion": cm,
        "iou": iou,
        "dice": dice_per_class(cm),
        "mean_iou": float(np.nanmean(iou)),
        "accuracy": float(np.diag(cm).sum() / max(cm.sum(), 1)),
        "names": names,
    }


def report(res, title="results"):
    names, cm = res["names"], res["confusion"]
    print(f"\n{title}")
    print(f"  mean IoU {res['mean_iou']:.3f}    pixel accuracy {res['accuracy']:.3f}"
          f"   <- accuracy is the misleading one")
    print(f"  {'class':<22}{'IoU':>8}{'Dice':>8}{'truth px':>12}{'pred px':>10}")
    for i, n in enumerate(names):
        iou = res["iou"][i]
        dice = res["dice"][i]
        shown_iou = "   absent" if np.isnan(iou) else f"{iou:8.3f}"
        shown_dice = "        " if np.isnan(dice) else f"{dice:8.3f}"
        flag = "  <- never predicted" if cm[:, i].sum() == 0 and cm[i].sum() > 0 else ""
        print(f"  {n:<22}{shown_iou}{shown_dice}{cm[i].sum():12,}{cm[:, i].sum():10,}{flag}")


def report_confusion(res):
    names, cm = res["names"], res["confusion"]
    width = max(len(n) for n in names) + 1
    print(f"\n  confusion, row = truth, column = prediction, % of each true class")
    print("  " + " " * width + "".join(f"{n[:9]:>10}" for n in names))
    for i, n in enumerate(names):
        total = max(cm[i].sum(), 1)
        print(f"  {n:<{width}}" + "".join(f"{100*cm[i, j]/total:9.1f}%" for j in range(len(names))))


# ---------- the free baselines any model has to beat ----------

def baseline_constant(y_true, cls, names):
    return summarise(y_true, np.full_like(y_true, cls), names)


def baselines(y_train, y_test, names):
    """Trained on nothing but the training labels, so they are honest
    comparators rather than cheats."""
    majority = int(np.bincount(y_train.ravel(), minlength=len(names)).argmax())
    out = {f"always {names[majority]}": baseline_constant(y_test, majority, names)}
    # Per-pixel prior: the most common class at each pixel position across the
    # training set. Beats "always background" whenever the photos are framed
    # alike, which is exactly the shortcut a weak model would learn.
    counts = np.stack([(y_train == c).sum(axis=0) for c in range(len(names))])
    out["per-pixel prior"] = summarise(y_test, counts.argmax(axis=0)[None].repeat(len(y_test), 0), names)
    return out
