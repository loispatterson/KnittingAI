"""Deciding which images the model is allowed to learn from.

The original split was X[:int(len(X)*0.8)] against sorted filenames. Sorted
filenames group by stitch type, so the last 20% was a single stitch -- the
model was validated on garter having trained mostly on knit and purl, and the
score said more about the alphabet than the model.

Shuffling fixes that but introduces a worse problem. knit11.jpg and knit9.png
are 0.96 correlated: the same swatch, saved twice. A random split can put one
in training and the other in test, and the test score then measures memory
rather than generalisation -- the same mistake as a random split on satellite
tiles, where neighbouring pixels are near-copies of each other.

So images are grouped first, and whole groups move together. With 29 images a
single test set of six is too small to trust, so the default is k-fold: every
image is in the test set exactly once, and the spread across folds is reported
alongside the mean. A wide spread is itself the finding -- it says the dataset
is too small to support the claim, which is true here and worth saying.
"""
import numpy as np

# Groups of images that must never be split across train and test. Keyed by
# filename stem. Near-duplicates found by correlating downsampled images -- see
# the note above; rerun that check when new photos are added.
DUPLICATE_GROUPS = [
    {"knit11", "knit9"},
]


def group_of(stem):
    """The unit that moves together. A near-duplicate pair is one unit; every
    other image is its own, because garter1 and garter2 are different swatches
    and keeping them apart would waste what little data there is."""
    for g in DUPLICATE_GROUPS:
        if stem in g:
            return "dup:" + "+".join(sorted(g))
    return stem


def k_folds(stems, k=5, seed=0):
    """Yields (train_idx, test_idx) k times, grouped so no group straddles."""
    groups = {}
    for i, s in enumerate(stems):
        groups.setdefault(group_of(s), []).append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    for fold in range(k):
        test_keys = keys[fold::k]
        test = sorted(i for key in test_keys for i in groups[key])
        train = sorted(i for key in keys if key not in set(test_keys) for i in groups[key])
        yield train, test


def describe(stems, y, names, train, test):
    """Which classes each side of a split actually contains. A test fold with no
    garter pixels cannot say anything about garter, and the IoU table will show
    it as absent rather than as a zero."""
    def shares(idx):
        sub = y[idx]
        return [float((sub == c).mean()) for c in range(len(names))]
    tr, te = shares(train), shares(test)
    print(f"    train {len(train):>2} images, test {len(test):>2} images")
    for c, n in enumerate(names):
        warn = "  <- absent from test" if te[c] == 0 else ""
        print(f"      {n:<22} train {tr[c]:6.2%}   test {te[c]:6.2%}{warn}")
