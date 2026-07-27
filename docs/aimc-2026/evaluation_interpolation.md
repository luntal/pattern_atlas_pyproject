# Interpolation Evaluation

## Context
This document describes the interpolation evaluation process for the aimc-2026 branch, reports the current metrics, and explains the interpolation accuracy behavior across positions.

## Run Configuration
- Branch: aimc-2026
- Model path: /models/aimc-2026/model_2026-04-08_09-52-22.pt
- Dataset path: /data/aimc-2026/dataset-groove.npz
- Selected samples: 485
- Interpolation pairs: 20
- Interpolation steps between endpoints: 10
- Interpolation positions per pair: 12 (start + 10 intermediate + end)

## Tolerances Used
- Position tolerance: +/-6 token positions
- Pitch tolerance: +/-1 pitch step
- Velocity tolerance: +/-30% of the velocity token range
- Micro timing tolerance: +/-20% of the timing token range

## Reported Results
- Mean overall vs original A (all positions): 0.7177
- Mean overall vs original B (all positions): 0.7245
- Mean overall vs closer original (all positions): 0.8084
- Mean pitch vs closer original (all positions): 0.6599
- Mean velocity vs closer original (all positions): 0.8651
- Mean micro timing vs closer original (all positions): 0.8452

The generated plot is saved at:
- docs/aimc-2026/evaluation_interpolation.png

## Evaluation Method
For each interpolation pair:
1. Encode the two endpoint sequences.
2. Interpolate in latent space with spherical interpolation.
3. Decode each interpolation position into a token sequence.
4. Compare each generated sequence against both originals (A and B).
5. Compute category-aware windowed accuracy:
   - Match within the allowed position window.
   - Accept category-specific value tolerances (pitch/velocity/timing).
   - Allow each predicted token to be used at most once.

Then aggregate over all pairs.

## Metric Formula
For each metric:

$$
\text{accuracy} = \frac{\text{matched reference tokens}}{\text{reference token count}}
$$

Reference tokens are always taken from the compared original sequence (A or B).

## Why The Minimum Interpolation Accuracy Can Average Around 0.6
A dip around the middle interpolation positions is expected because those sequences are furthest from both endpoints.

Key reasons:
- Midpoint blending effect: center positions are a latent blend, so they are less similar to either original A or original B than endpoint-near positions.
- Single-endpoint comparison lines: the plotted lines x vs original A and x vs original B each compare against one fixed endpoint, so both typically dip near the center.
- Partial structural preservation: rhythmic/token structure is still partly preserved, and tolerances allow near-matches, so accuracy does not collapse to zero.
- Averaging across 20 pairs: local lows from some pairs are smoothed by easier pairs, making the minimum of the averaged curve higher and often around a moderate floor (such as about 0.6).

So a minimum near 0.6 in the averaged interpolation curve is consistent with successful interpolation that preserves part of the rhythmic/token structure while moving away from each endpoint in the latent space.
