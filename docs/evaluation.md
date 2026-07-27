# Reconstruction Quality Evaluation

## Summary
This short evaluation compares original token sequences against model-generated sequences in direct inference mode (no interpolation). The objective is to assess reconstruction fidelity at both full-sequence and token-property levels.

The latest run reports strong reconstruction behavior, with high sequence-level agreement and low reconstruction loss.

## Experimental Context
- Model checkpoint: `models/global-grooves/model_best_train_2026-04-27_15-21-24.pt`
- Dataset: `data/global-grooves/dataset-merged-test.npz`
- Evaluated sample count: 18 sequences

## Data Split and Objectivity Note
The rhythms used for this evaluation are sampled from the validation subset, which is created by splitting the full available dataset into training and validation partitions. This supports a realistic and objective perspective because validation examples are not used for parameter updates during training.

At the same time, validation samples still come from the same underlying corpus as the training split, so they can resemble training data in style and token statistics. Therefore, the reported reconstruction metrics should be interpreted as in-distribution generalization performance rather than performance on a fully out-of-distribution test set.

## Metrics and Definitions
- Exact match (%): Percentage of sequences where the generated sequence matches the original sequence exactly (after alignment/padding rules).
- Token accuracy: Fraction of correctly predicted tokens over all non-padding tokens.
- Cross entropy reconstruction loss: Mean token-level negative log-likelihood under teacher-forced reconstruction logits.
- Pitch accuracy: Accuracy restricted to pitch-token positions.
- Velocity MAE: Mean absolute error on velocity-token values. Velocity is quantized into 32 bins, so MAE ≈ 3 means average error is about 3 bins, which represents about 12 to 13 MIDI velocity units on average.
- Micro Timing MAE: Mean absolute error on micro-timing-token values.

## Results Overview
- Exact match (% identical sequences): **94.444%**
- Token accuracy: **0.943454**
- Cross entropy reconstruction loss: **0.006980**
- Pitch accuracy: **0.934114**
- Velocity MAE: **2.840410**
- Micro Timing MAE: **3.404099**

## Interpretation
The model reconstructs rhythmic patterns with high overall fidelity. The high exact-match rate and token accuracy indicate strong structural preservation, while low cross-entropy supports confident token reconstruction. Pitch accuracy remains high, and the moderate velocity/micro-timing MAE values suggest that most residual differences are concentrated in expressive dimensions rather than coarse rhythmic structure.

## Interpolation Quality

This section reports interpolation quality averaged over 10 random anchor pairs using 10 interpolation steps. Similarity is token-level match ratio to each anchor.

### How Similarity Is Calculated

For each interpolated generated sequence at index $i$ and each anchor sequence $A$:

1. The generated token sequence is aligned to the anchor length (trim or pad with `pad_token`).
2. Only non-padding anchor positions are evaluated.
3. Similarity is computed as the fraction of exact token matches on these valid positions.

Formally, for anchor $A$ and generated sequence $G_i$:

$$
	ext{Similarity}(A, G_i) = \frac{\sum_t \mathbf{1}[A_t = G_{i,t}]\,\mathbf{1}[A_t \neq \text{pad}]}{\sum_t \mathbf{1}[A_t \neq \text{pad}]}
$$

The reported table values are then averaged across all sampled anchor pairs at each interpolation index.

### Why Averaged Similarity Often Stays Below 0.5

- The table reports per-index averages over many different anchor pairs, not best-case examples.
- Rhythm token sequences are long and sparse in exact token agreement across unrelated pairs, so exact token-match ratios are naturally moderate.
- Middle interpolation points are intentionally between anchors, so they are not expected to strongly match either endpoint.
- Even at endpoint-near indices, averaging over many pair difficulties lowers the global mean.
- This is a strict exact-token metric (not perceptual similarity), so small token shifts reduce score substantially.

### Averaged Interpolation Quality Metrics (30 random pairs)
mean_sim_a0        : 0.146258 +- 0.073271
mean_sim_a1        : 0.138770 +- 0.055700
start_a0_advantage : 0.299757 +- 0.273032
end_a1_advantage   : 0.327417 +- 0.313326
trend_slope        : 0.051727 +- 0.032199
monotonic_fraction : 0.684848 +- 0.116736

### Per-Index Averages

```text
 Idx | Similarity->A0 | Similarity->A1 | Closer Anchor
  0 |       0.361843 |       0.062086 | A0
  1 |       0.314843 |       0.058237 | A0
  2 |       0.228392 |       0.069259 | A0
  3 |       0.198719 |       0.069241 | A0
  4 |       0.134385 |       0.068118 | A0
  5 |       0.095793 |       0.087217 | A0
  6 |       0.080888 |       0.091297 | A1
  7 |       0.074323 |       0.113206 | A1
  8 |       0.069854 |       0.146995 | A1
  9 |       0.066793 |       0.207243 | A1
 10 |       0.064926 |       0.300583 | A1
 11 |       0.064335 |       0.391752 | A1
```

Compact interpretation: interpolation trajectories show a clear directional transition from A0-dominant similarity at early indices to A1-dominant similarity at late indices.
