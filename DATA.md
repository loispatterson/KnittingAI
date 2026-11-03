# Where the images came from, and what that means

## The honest position

The 29 image/mask pairs in `data/` were collected in 2025 while learning, and
their provenance was not recorded at the time. None of them carry camera
metadata, and the sizes give them away: 275x183, 324x156, 259x194, 225x225 are
web thumbnail dimensions, and `test_knit.jpg` reports Canva as its software.
They were, in all likelihood, downloaded rather than photographed.

So they are not mine to license, this repository is public, and they are in its
history as well as its working tree. That is a mistake rather than a position,
and it is recorded here rather than quietly fixed, because the fix changes the
dataset and the dataset is the part of a small machine-learning project that
most affects the result.

The masks are my own work: hand-painted, one per image, in the colours listed
in `color_map.json`.

## What replaces them

Photographs of my own swatches, taken for this purpose. That solves the
provenance question outright, and it also happens to be what the model needs:
the confusion matrix shows stockinette and reverse stockinette -- the two faces
of the same fabric -- being mistaken for each other in roughly a third of
pixels, and those are the two classes with the least data relative to how hard
they are to tell apart.

A replacement set should record, per image: who took it, when, what the stitch
is, the yarn weight and needle size, and the approximate gauge. Gauge in
particular is not optional once the product is a pattern rather than a mask --
a stitch count means nothing without it.

## The other licensing question: patterns

The eventual product is "photograph a jumper you like, get a pattern you can
knit". Two things are worth being clear about, neither of which is legal
advice.

A written knitting pattern is copyrighted as a literary work -- its text,
charts, photographs and layout. Stitch structures are not: garter, stockinette,
rib and cable are techniques, many of them centuries old, and nobody owns them.
Working out your own instructions by looking at a garment does not copy anyone's
text. Where it gets less clear is surface decoration -- a Fair Isle motif or an
intarsia picture can be an artistic work in its own right -- and UK and EU
design rights can protect a garment's appearance in ways US law largely does
not. The photograph itself belongs to whoever took it.

The second thing matters more in practice. Independent designers earn very
little and the knitting community polices reverse-engineering hard, quite apart
from the law.

So this tool is deliberately scoped to describe **fabric**, not to reproduce
**designs**: which areas are which stitch, and at what gauge. That is a
knitting-literacy instrument and it is what the model actually does. Emitting
"here is that designer's jumper, reconstructed" would be the same pipeline
pointed at a different purpose, and it is not the purpose.
