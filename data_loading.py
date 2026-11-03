"""Loading the knitting images and their hand-painted masks.

This file exists because the original loader was wrong in three ways that no
training log could have shown, and the fixes are worth keeping separate from
the model so they can be tested.

  1. It paired images with masks by position in two sorted lists. The "_mask"
     suffix changes the sort order -- an underscore sorts after digits, so
     images went knit1, knit10, knit11 while masks went knit10, knit11, knit1.
     Five of 29 pairs were wrong. Pairing is now by filename stem, and a stem
     without a partner is an error rather than a silent shift.

  2. It assigned any pixel whose colour was not in the map to class 0. The map
     had garter as (61,245,71) and the masks are painted (61,245,61), so all
     seven garter masks loaded as pure background and the model never saw the
     class. Unmapped pixels are now counted and reported, and anything above a
     small tolerance stops the run.

  3. It did not scale the pixel values, while predict.py divided by 255. The
     model trained on 0-255 and was asked to predict on 0-1. Scaling now
     happens in one place, here, and nowhere else.
"""
import json
import os

import cv2
import numpy as np

IMG_SIZE = 128
# How much of a mask may be unrecognised before we refuse to train on it. Hand
# painting leaves a thin halo of blended colour on every boundary, so this
# cannot be zero -- but 11% across the set was a typo, not a halo.
UNMAPPED_TOLERANCE = 0.02


def load_color_map(path="color_map.json"):
    """{"garter": {"(61, 245, 61)": 2}} -> {(61,245,61): 2}, ["background", ...]

    The labels come back ordered by class id rather than by position in the
    file, because the list is indexed by prediction and the file is not
    guaranteed to be in order.
    """
    data = json.load(open(path))
    colours, names = {}, {}
    for name, entry in data.items():
        for colour_str, idx in entry.items():
            colours[tuple(int(x) for x in colour_str.strip("() ").split(","))] = idx
            names[idx] = name
    missing = set(range(len(names))) - set(names)
    if missing:
        raise ValueError(f"class ids are not contiguous from 0: missing {sorted(missing)}")
    # Classes may legitimately have no pixels yet: the map is the plan, the
    # masks are the progress against it. The model still builds an output
    # channel per class, and evaluate.py reports an unpainted class as absent
    # rather than as a zero, so adding its masks later needs no code change.
    return colours, [names[i] for i in sorted(names)]


def _pairs(image_dir, mask_dir):
    """Match images to masks by filename stem, not by sort position."""
    images = {os.path.splitext(f)[0]: f
              for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))}
    masks = {os.path.splitext(f)[0].removesuffix("_mask"): f
             for f in os.listdir(mask_dir) if f.endswith((".jpg", ".png"))}
    only_image = sorted(set(images) - set(masks))
    only_mask = sorted(set(masks) - set(images))
    if only_image:
        print(f"  no mask for: {', '.join(only_image)}")
    if only_mask:
        print(f"  no image for: {', '.join(only_mask)}")
    return [(stem, images[stem], masks[stem]) for stem in sorted(set(images) & set(masks))]


def load_data(image_dir="data/images", mask_dir="data/masks",
              color_map=None, img_size=IMG_SIZE, verbose=True):
    """Returns X (n, size, size, 3) float32 in 0-1, y (n, size, size) uint8, names."""
    if color_map is None:
        color_map, _ = load_color_map()

    pairs = _pairs(image_dir, mask_dir)
    X, y, stems, problems = [], [], [], []

    for stem, img_name, mask_name in pairs:
        img = cv2.imread(os.path.join(image_dir, img_name))
        mask = cv2.imread(os.path.join(mask_dir, mask_name))
        if img is None or mask is None:
            raise OSError(f"could not read the pair for {stem}")

        img = cv2.resize(img, (img_size, img_size))
        # INTER_NEAREST on the mask: any interpolation invents colours that are
        # in no class, which is how label noise gets manufactured by the loader.
        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        labels = np.zeros(mask.shape[:2], dtype=np.uint8)
        matched = np.zeros(mask.shape[:2], dtype=bool)
        for colour, idx in color_map.items():
            hit = (mask == colour).all(axis=-1)
            labels[hit] = idx
            matched |= hit

        unmapped = float((~matched).mean())
        if unmapped > UNMAPPED_TOLERANCE:
            problems.append((stem, unmapped, _worst_unmapped(mask, matched)))

        X.append(img.astype(np.float32) / 255.0)
        y.append(labels)
        stems.append(stem)

    if problems:
        print("\nMasks with colours that are in no class:")
        for stem, share, colour in problems:
            print(f"  {stem:<22} {share:6.1%} unmapped, mostly RGB {colour}")
        raise ValueError(
            f"{len(problems)} mask(s) above the {UNMAPPED_TOLERANCE:.0%} tolerance. "
            "Those pixels would silently become background, which is how a whole "
            "class disappears. Fix color_map.json or repaint the mask."
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.uint8)
    if verbose:
        print(f"\nloaded {len(X)} pairs at {img_size}x{img_size}, "
              f"pixels in [{X.min():.2f}, {X.max():.2f}]")
        unpainted = sorted(set(color_map.values()) - set(np.unique(y).tolist()))
        if unpainted:
            _, names = load_color_map()
            print("  classes in the map with no pixels painted yet: "
                  + ", ".join(f"{names[i]} ({i})" for i in unpainted))
    return X, y, stems


def _worst_unmapped(mask, matched):
    """The most common colour that matched nothing -- usually names the typo."""
    stray = mask[~matched].reshape(-1, 3)
    if not len(stray):
        return None
    colours, counts = np.unique(stray, axis=0, return_counts=True)
    return tuple(int(v) for v in colours[counts.argmax()])


def class_balance(y, names):
    """Pixel share per class. Printed before training because it decides which
    metrics mean anything: at 69% background, accuracy is not a metric."""
    rows = []
    total = y.size
    for idx, name in enumerate(names):
        n = int((y == idx).sum())
        rows.append((idx, name, n, n / total))
    return rows
