# KnittingAI

Reading knitted fabric from a photograph: which areas are stockinette, which
are garter, which are the reverse side — and eventually, at what gauge, so the
result is something you can actually knit.

A small U-Net, 29 hand-painted masks, and an evaluation harness that is larger
than the model. The harness is the point. See [DATA.md](DATA.md) for the
image provenance and where this stops short of reproducing anyone's design.

## The state of it

Five-fold cross-validation, grouped so that near-duplicate photographs cannot
straddle a split:

```
  baseline  always background      mean IoU 0.161
  baseline  per-pixel prior        mean IoU 0.239
  model                            mean IoU 0.496   folds [0.33, 0.48, 0.60, 0.59, 0.48]
```

So the model is roughly twice the best free baseline — and the spread across
folds is 0.27, more than half the score. The honest reading is "about 0.5, give
or take 0.13", and the binding constraint is not the architecture. Twenty-nine
images cannot support a more precise claim than that.

Pooled over all folds, per class:

```
  background            IoU 0.778
  garter                IoU 0.467
  stockinette           IoU 0.280
  reverse_stockinette   IoU 0.232
```

And the confusion that matters: stockinette is called reverse stockinette 27%
of the time, reverse stockinette is called stockinette 42% of the time. Those
are the two faces of the same fabric, and telling them apart *is* the product —
a ragged boundary washes out when pixels are aggregated into stitches, but
knit-for-purl is the pattern being wrong.

## Seven things that were wrong, and how each was invisible

This project reported `accuracy` on a segmentation where background is 58% of
pixels. Predicting "background" everywhere scores 48.7%. Every bug below was
compatible with a training log that looked like it was working.

1. **A one-digit typo deleted a class.** `color_map.json` had garter as
   `(61, 245, 71)`; the masks are painted `(61, 245, 61)`. Unmatched pixels
   silently fell through to class 0, so all seven garter masks loaded as pure
   background — `garter7_mask.png` is 87% garter, and none of it reached the
   model. Accuracy *improves* when a minority class vanishes.

2. **Images were paired with the wrong masks.** The loader zipped two sorted
   lists, and `_mask` changes the sort order because an underscore sorts after
   digits: images ran `knit1, knit10, knit11`, masks ran `knit10, knit11,
   knit1`. Five of 29 pairs were wrong — 17% of the data teaching the opposite
   of the truth.

3. **Train/serve skew.** Training never divided by 255; `predict.py` did. The
   model learned on 0–255 and was asked to predict on 0–1.

4. **`NUM_CLASSES = 4` against a seven-class map.** Three classes could not be
   predicted at any threshold.

5. **Accuracy as the metric**, on data where a constant beats most mistakes.

6. **An unshuffled split.** `X[:int(len(X)*0.8)]` over sorted filenames means
   validating on whichever stitch sorts last — one class, not a sample.

7. **Leakage waiting to happen.** `knit11.jpg` and `knit9.png` correlate at
   0.96: the same swatch twice. A random split puts one each side and measures
   memory instead of generalisation — the same mistake as a random split on
   satellite tiles, where neighbouring pixels are near-copies.

## Layout

```
data_loading.py   pairing by filename stem, colour→class, scaling, and a hard
                  stop when too much of a mask matches no class
evaluate.py       IoU and Dice per class, confusion matrix, and the free
                  baselines any model has to beat before its score means
                  anything
splits.py         grouped k-fold, so near-duplicates move together
train.py          the U-Net, flip augmentation, class weighting, early
                  stopping on held-out mean IoU
color_map.json    the plan: seven classes, four painted so far
plot_pattern.py   mask → stitch grid (see the open question below)
TinyUNet.py       the original single-split script, kept for comparison
```

```sh
python train.py           # 5-fold, the number worth quoting
python train.py --final   # train on everything and save
```

## Open

**More photographs, of my own swatches.** Both the licensing fix and what the
confusion matrix asks for. Stockinette and reverse stockinette first.

**A pretrained encoder.** Sixteen filters learned from scratch on 23 images is
not where the information is.

**Stitch pitch.** `plot_pattern.py` currently draws the 128×128 mask as though
each pixel were a stitch. It is not: a photograph might hold 40 stitches across
those pixels. Knitted fabric is near-periodic, so the dominant spatial
frequency in each direction gives stitches-per-pixel — and without that there
is no gauge, and without gauge there is no pattern. This is the step between a
segmentation demo and something a knitter could use.

**Classes 4 and 5.** Garter is alternating knit and purl rows, so those may be
derived from a garter region by `garter_to_knit_purl` rather than segmented
directly — which needs no new masks. Worth deciding on purpose rather than by
default.
