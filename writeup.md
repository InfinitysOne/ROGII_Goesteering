# ROGII Wellbore Geology Prediction — Solution Writeup

> This document is structured around the 5 evaluation criteria. Sections are pre-filled
> with what can be determined from the current codebase (`dataset.py`, `model.py`,
> `train_full.py`, `inference_all_wells.ipynb`). Anything marked **[TODO]** needs input
> from you — it can't be reconstructed from the code alone, either because it depends on
> experiments not present in these files, or because the mechanism doesn't exist yet.

---

## 1. Breadth and Depth of Exploration

### Approach A: CNN + Bidirectional LSTM hybrid on (GR, Z) sliding windows

**Idea and motivation**
Predict True Vertical Thickness (TVT) at each measured-depth (MD) position from a local
window of the last `WINDOW_SIZE=50` ft of two logs: Gamma Ray (GR) and depth-track Z.
The architecture combines a 2-layer 1D CNN (16→32 channels, kernel size 3, max-pooled
between layers) to extract local shape features from the log curves, followed by a
single-layer bidirectional LSTM (hidden size 64, so 128-dim after concatenation) to
capture sequential/positional structure across the window, and a small fully-connected
head (128→64→1) with dropout (p=0.2) for the final regression.

Rationale: GR responds to lithology changes (shale vs. sand/carbonate), and its local
pattern shape (spikes, trends, cyclicity) is expected to correlate with structural
position relative to the target zone; the CNN is meant to pick up these local textures,
while the LSTM is meant to integrate them across the window to track drift/trend.

**Data handling**
- Wells split at the **well level** (80/20, `random_state=42`), not row level — this
  correctly prevents leakage between train and validation, since the model would trivially
  memorize interpolated values on the depth axis if rows from the same well went to both splits.
- GR gaps forward/backward filled; TVT rows with missing ground truth dropped for training.
- Fixed-window slicing: every position `i` produces one training sample
  `(features[i:i+50], TVT[i+49])` — i.e. a many-to-one regression per window, not
  many-to-many; the model never predicts a full trajectory in one pass.

**Validation results**
`[TODO — paste actual numbers from your training logs]`
Example format to fill in:
| Epoch | Train RMSE (ft) | Val RMSE (ft) |
|-------|------------------|----------------|
| 1     |                  |                |
| ...   |                  |                |
| Best  |                  |                |

**Conclusions / lessons learned**
`[TODO]` — e.g.: Did val RMSE plateau early? Did train/val gap suggest over/underfitting?
Did specific wells dominate the validation error? Any qualitative pattern in the inference
plots (`img/inference_plot_*.png`) about where predictions diverge from ground truth —
near faults, at the start of the blind-flying region, in wells with sparse GR data, etc.?

### Approach B: `[TODO — name of second approach]`

The rubric explicitly wants **multiple genuinely different approaches** (different feature
sets, modeling strategies, or methodological choices — not just window-size or hyperparameter
tweaks). The current codebase only contains one architecture. If you tried, for example:
- a pure gradient-boosted-tree baseline on hand-engineered features,
- a transformer/attention-based sequence model,
- a physics-informed approach (e.g. incorporating dip angle or structural continuity constraints),
- feature ablations (GR only vs. GR+Z vs. additional logs),

...document each with the same structure (idea → validation numbers → what you learned,
including *why* it under/over-performed relative to Approach A). If you only ran the one
architecture in this repo, be explicit that this is a single deeply-explored approach rather
than implying breadth that isn't there — the rubric rewards honest depth over inflated breadth.

---

## 2. Insights About the Data and Wells

`[TODO]` — this section needs your own observations; the code doesn't currently log or
surface data diagnostics. Things worth checking and reporting, if you haven't already:

- **GR data quality across wells**: how much of the GR curve required fill (`ffill`/`bfill`)
  per well? Wells with large missing stretches likely have less reliable predictions —
  worth quantifying and cross-referencing against per-well validation error.
- **TVT range vs. the hardcoded normalization** (`tvt_mean=11000`, `tvt_std=2000`): do the
  actual training wells' TVT values cluster near this assumption, or are there wells far
  outside this range that get poorly scaled?
- **Blind-flying region behavior** (visible in the inference notebook's red-shaded region):
  does prediction error grow with distance from the last known TVT point? This is expected
  physically (uncertainty compounds the further you extrapolate without a new tie-point) —
  worth checking directly, since it also feeds into Section 5 (uncertainty).
- **Cross-well variability**: were there wells where the model failed badly? Any structural
  reason (faulted section, unusual geology, sensor issue)?
- **How much did the public dataset generalize** to your validation wells — i.e. did held-out
  wells behave similarly to training wells, or were there systematic differences?

---

## 3. Physical Meaningfulness of the Solution

**What is physically grounded here:**
- Well-level (not row-level) train/val split respects the fact that geology is spatially
  correlated within a well — this avoids a metric-optimizing shortcut (interpolation) that
  would not reflect real predictive skill.
- Using GR as a primary feature is physically motivated: gamma ray response is a standard,
  well-understood lithology indicator in geosteering, not an arbitrary correlate.
- The windowed/local approach (50 ft context) matches the physical intuition that geosteering
  decisions are made from a local rolling window of recent log data, not the full well history —
  this mirrors how the problem is actually solved in the field.

**Where this leans toward metric optimization rather than physical modeling:**
`[TODO — be honest here, this is explicitly what the rubric wants reflected on]`
Some candidate points to address:
- The model treats TVT prediction as a black-box regression from raw curve shape; it does
  not explicitly model bed dip, structural geometry, or fault offsets, so it can't
  distinguish "the log looks like this because of true geological change" vs. "the log looks
  like this by coincidence" — a purely physically-grounded model would encode more structural
  constraints (e.g. continuity, monotonic depth relationships within a bed).
- Normalization constants (`150.0`, `10000.0`, `11000.0`, `2000.0`) are fixed heuristics, not
  derived from the physical range of the target formation — if pushed hard enough, hyperparameter
  or normalization tuning against validation score risks overfitting to the specific well set
  rather than reflecting genuine subsurface physics.
- `[TODO]` — Did you consider or reject any features/ensembling purely because they improved
  leaderboard/validation score without a clear geological justification? Documenting that
  reasoning (what you *didn't* do and why) is exactly what this section wants.

---

## 4. Contribution of Individual Ideas

`[TODO]` — the rubric wants **quantified** ablations. Right now the codebase trains one
final configuration; there's no ablation harness. Suggested minimum set of experiments to
run and report as a table:

| Change | Val RMSE (ft) | Δ vs. baseline |
|---|---|---|
| GR only (no Z) | | |
| GR + Z (current) | | |
| CNN only (remove LSTM) | | |
| LSTM only (remove CNN) | | |
| Unidirectional vs. bidirectional LSTM | | |
| Window size 25 / 50 / 100 | | |
| With / without dropout | | |
| Fixed normalization vs. per-well/train-set-derived normalization | | |

Even 3–4 of these rows, run once each with the same seed and val split, would let you make
a defensible, quantified claim like "removing the LSTM increased val RMSE by X ft, showing
sequence context contributes meaningfully beyond local CNN features" — which is exactly the
kind of evidence this section is scored on. Currently `train_full.py` doesn't expose a way to
toggle these components without editing `model.py` directly — worth adding a config flag if
you're going to run this ablation sweep.

---

## 5. Uncertainty Estimation

**Current state: none.** This is the clearest gap in the current pipeline. The model outputs
a single deterministic point estimate (`tvt_prediction.squeeze(-1)`), and the inference
notebook's only handling of failure is a hard fallback to the global mean when a prediction
is `NaN` — this is a data-completeness patch, not an uncertainty estimate, and it should not
be presented as one in the writeup.

**Options to add real uncertainty, roughly in order of implementation cost:**

1. **MC Dropout** (cheapest): keep `model.train()` (so dropout stays active) at inference
   time, run each window through the model N times (e.g. 20–50), and report the mean and
   std of the predictions as your point estimate and confidence interval. Requires no
   retraining — just a change to the inference loop.
2. **Deep ensemble**: train the existing architecture k times (different seeds/data order),
   average predictions for the point estimate, use inter-model variance as uncertainty.
   More expensive (k× training time) but generally better-calibrated than MC dropout.
3. **Quantile/distributional regression head**: replace the single-output regression with
   either a pinball-loss quantile head (predict e.g. 10th/50th/90th percentile) or a
   Gaussian-NLL head (predict mean and variance directly), trained end-to-end.
4. **Physically-motivated uncertainty proxy**: since error is expected to grow with distance
   from the last known TVT tie-point (blind-flying distance) and with GR data gaps, a cheap
   complementary signal is to track "distance since last known TVT" / "fraction of window
   that was filled rather than measured" per prediction and report it alongside the point
   estimate, even without a fully learned uncertainty head.

**Reliability discussion (fill in once one of the above is implemented):**
`[TODO]` — where does the model do well vs. poorly? Concretely: does predicted uncertainty
increase near faults / in the blind-flying region / in wells with heavy GR fill? Does actual
error correlate with the reported uncertainty (a calibration check — e.g. do ~90% of true
values fall inside your reported 90% interval)? This is the evidence the rubric is asking for
under "how this uncertainty is quantified and communicated."

---

## Summary / Honest Assessment

`[TODO — 2-3 sentences]` What's implemented is a single, physically-reasonable, well-split-respecting
CNN-BiLSTM baseline with no uncertainty quantification and no documented ablations yet. The
biggest gaps against the rubric right now are: (1) only one approach is present — no comparative
exploration; (2) no quantified contribution analysis; (3) no uncertainty estimation at all.
These are the three highest-leverage things to add before finalizing the writeup.
