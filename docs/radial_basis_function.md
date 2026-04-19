# RBF Heatmap Modes: Mag and Conf

This document explains the two heatmap modes used in the interactive RBF viewer in [src/rbf.py](../src/rbf.py).

## Overview

The viewer computes an interpolated latent vector at each 2D position $(x, y)$.
The heatmap can visualize either:

1. `Mag` (magnitude): how large the interpolated latent vector is.
2. `Conf` (confidence): how strongly the point is supported by nearby anchors.

Both modes are normalized to $[0, 1]$ per redraw for visual contrast.

## Mag Mode

`Mag` computes the $L_2$ norm of the interpolated latent vector.

For a point $(x, y)$, let $z(x, y)$ be the RBF-interpolated latent vector. Then:

$$
\mathrm{Mag}(x, y) = \|z(x, y)\|_2
$$

Interpretation:

1. Bright regions: large latent amplitude.
2. Dark regions: small latent amplitude.

This mode answers: "How strong is the interpolated vector here?"

## Conf Mode

`Conf` does not depend on latent-vector values directly. It uses the sum of Gaussian RBF basis responses from all control points.

If control points are $p_i$ and RBF width is $\sigma$, then:

$$
\mathrm{Conf}(x, y) = \sum_i \exp\left(-\frac{\|[x, y] - p_i\|^2}{2\sigma^2}\right)
$$

Interpretation:

1. Bright regions: dense support from nearby anchors.
2. Dark regions: sparse support, farther from anchors.

This mode answers: "How well supported is interpolation at this location?"

## Normalization Detail

After computing the raw heatmap for either mode, the viewer applies min-max normalization:

$$
H_{\text{norm}} = \frac{H - H_{\min}}{H_{\max} - H_{\min}}
$$

If $H_{\max} = H_{\min}$, the viewer falls back to zeros.

Practical implication:

1. Colors are best for relative comparison within the current frame.
2. Absolute color values are not directly comparable across different settings (different $\sigma$, anchor positions, mode, or grid resolution).

## Which Mode To Use

1. Use `Mag` to inspect output strength in latent space.
2. Use `Conf` to inspect spatial support and interpolation reliability.

In short:

1. `Mag` is output-centric.
2. `Conf` is support-centric.

## Influence Diagram

The influence diagram (bar chart) shows how much each anchor point influences the current cursor position.

X-axis:

1. Anchor index (one bar per anchor point).

Y-axis:

1. Influence strength.

The diagram supports two modes: raw influence and percentage influence.

### Raw Per-Anchor Influence

For cursor position $x = [x, y]$ and anchor position $p_i$, the raw influence of anchor $i$ is:

$$
\phi_i(x) = \exp\left(-\frac{\|x - p_i\|^2}{2\sigma^2}\right)
$$

Interpretation:

1. Values are in $(0, 1]$.
2. Anchors closer to the cursor have larger influence.
3. Anchors far away have influence close to 0.

### Percentage Influence

Percentage influence is the normalized share of each anchor relative to all anchors:

$$
\alpha_i(x) = \frac{\phi_i(x)}{\sum_k \phi_k(x)}
$$

Interpretation:

1. Each bar is a fraction in $[0, 1]$.
2. All bars sum to 1 (or 100%).
3. This makes it easy to read statements like:
	Anchor 1 contributes about 80%, Anchor 2 about 10%, etc.

### Simple Explanation (Non-Mathematical)

Imagine every anchor is a lamp and your cursor is a point on the floor.

1. Raw mode: shows how bright each lamp is at your current spot.
2. Percentage mode: shows how much of the total brightness comes from each lamp.

So:

1. Raw mode answers: How strongly is each anchor active here?
2. Percentage mode answers: How is total influence split among anchors?
