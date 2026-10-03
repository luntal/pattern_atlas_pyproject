# Reconstruction Evaluation

## Context
This document records the reconstruction evaluation run for the aimc-2026 branch and explains how each metric is computed.

## Run Configuration
- Branch: aimc-2026
- Model path: /models/aimc-2026/model_2026-04-08_09-52-22.pt
- Dataset path: /data/aimc-2026/dataset-groove.npz
- Selected samples: 100
- Compared sequence length: 218 tokens per sequence (minimum of predicted and target lengths)

## Tolerances Used
- Position tolerance: +/-24 token positions
- Pitch tolerance: +/-1 pitch step
- Velocity tolerance: +/-10% of the velocity token range
- Micro timing tolerance: +/-10% of the timing token range

## Reported Results
- Token accuracy (non-pad): 0.8625 (n = 14667)
- Pitch accuracy: 0.8223 (n = 3837)
- Velocity accuracy: 0.8658 (n = 3838)
- Micro timing accuracy: 0.8426 (n = 3838)

## Evaluation Method
The current evaluation uses a windowed position match, which is stricter than a pure order-invariant overlap but still tolerant to small shifts.

For each sequence:
1. Collect target token positions in a category.
2. Search the generated sequence within a limited position window around each target token.
3. Accept a match only if the token value also falls inside the category-specific tolerance.
4. Mark each predicted token as used at most once.
5. Count matched target tokens for that sequence.

Then aggregate across all selected sequences.

## Metric Formula
For any category:

$$
\text{accuracy} = \frac{\text{matched reference tokens}}{\text{reference token count}}
$$

where reference tokens are taken from the original target sequence.

For pitch tokens, the value must be within one pitch step of the reference token.
For velocity tokens, the value must be within 10% of the velocity token range.
For micro timing tokens, the value must be within 10% of the timing token range.

## Meaning of n
In each printed line, n is the denominator for that metric:
- Non-pad n: number of all non-pad target tokens evaluated.
- Pitch n: number of target tokens in the pitch token range.
- Velocity n: number of target tokens in the velocity token range.
- Micro timing n: number of target tokens in the micro timing token range.

So n is a token count, not a sequence count.

## Important Interpretation
- Token positions do not need to match exactly, but they must fall inside the allowed position window.
- Pitch is matched by absolute step distance, while velocity and micro timing are matched by percent of their token ranges.
- The overall token accuracy also uses these category tolerances, so pitch, velocity, and timing tokens are not scored by strict equality anymore.
- Each generated token can be used at most once.

## Notes
This evaluation reflects token-content quality within a limited temporal neighborhood, not strict step-by-step alignment in time.
The windowed tolerance is stricter than a fully order-invariant bag-of-tokens metric, but it still avoids the cascade effect where a single early insertion or omission would make many later tokens appear wrong.
Pitch uses an absolute step tolerance so small pitch shifts remain acceptable.
Velocity and micro timing use percentage-based tolerances so the accepted range scales with each token space.
