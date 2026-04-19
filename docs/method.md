# Latent-to-Memory Expansion

## Abstract
We present a Transformer-based symbolic rhythm VAE that conditions decoding through a sequence-structured latent memory. Instead of injecting a single latent vector directly into the decoder, we project each latent code into multiple learned memory tokens and use them as cross-attention memory at every autoregressive step. This design increases latent information accessibility during generation and improves reconstruction fidelity. We further combine this with per-dimension free bits and per-token KL normalization to stabilize latent usage and reduce collapse. Experiments on symbolic drum datasets show improved reconstruction quality and more coherent interpolations compared with single-vector latent conditioning baselines, while maintaining controllable latent traversals. These results suggest that latent-to-memory expansion is an effective conditioning strategy for sequence VAEs in symbolic music modeling.


## Related Work
Our decoder conditioning strategy is related to a broader line of work that expands compact latent codes into structured decoder inputs rather than injecting them only as a single global vector. In computer vision, Spatial Broadcast Decoder (Watters et al., 2019) showed that broadcasting latent variables into a structured representation can improve reconstruction and disentanglement, while earlier generator designs such as DCGAN (Radford et al., 2015) popularized latent projection followed by reshape into higher-dimensional feature tensors. In symbolic music, hierarchical latent approaches such as MusicVAE (Roberts et al., 2018) similarly demonstrated that decomposing a global latent representation into intermediate generation units improves long-range structure and reconstruction quality; related hierarchical systems include MIDI-Sandwich and MIDI-Sandwich2 (Liang et al., 2019a,b). Our method follows this general principle but adapts it to Transformer cross-attention by linearly projecting a single latent vector into a sequence of learned memory tokens, which are then queried autoregressively by the decoder. This can be viewed as a sequence-form latent expansion that is complementary to prior spatial broadcast and hierarchical latent decoding schemes.

## Method

### Overview
We propose a Transformer-based variational autoencoder for symbolic rhythm modeling with latent memory conditioning. Instead of conditioning the decoder with a single latent vector at each step, we transform each latent code into a sequence of learned memory slots and use them as cross-attention memory throughout autoregressive decoding. This design increases the effective latent bandwidth available to the decoder and improves latent utilization for reconstruction and interpolation.

### Input Representation
Each rhythm sequence is represented as a token sequence

$$
x = (x_1, \ldots, x_T)
$$

from a discrete vocabulary containing structural, pitch, velocity, and timing events. We use dedicated special tokens for padding, start-of-sequence, and end-of-sequence. In addition to token embeddings, we compute beat-progress tokens and embed them to provide bar-relative timing context.

### Encoder
The encoder maps an input sequence to a Gaussian posterior over latent variables. Token embeddings and beat embeddings are concatenated and projected to the model dimension. Positional embeddings are added, and a Transformer encoder processes the sequence with padding masks. We prepend a learnable classification token and use its final hidden state to predict posterior parameters:

$$
\mu = f_\mu(h_{\mathrm{CLS}}), \quad \log \sigma^2 = f_{\log \sigma^2}(h_{\mathrm{CLS}}).
$$

Latent samples are drawn via reparameterization:

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I).
$$

### Latent-to-Memory Projection
Our key architectural component is a latent expansion module that converts a global latent vector into a sequence-structured memory:

$$
z \in \mathbb{R}^{B \times d_z}
\rightarrow Wz + b \in \mathbb{R}^{B \times (N_m d)}
\rightarrow M \in \mathbb{R}^{B \times N_m \times d}.
$$

Here, $N_m$ is the number of memory slots and $d$ is the decoder model dimension. These memory slots are learned latent feature containers, not fixed timeline bins.

### Intuitive Interpretation of Latent Memory
A latent vector of length $d_z$ can be viewed in two equivalent ways: as a 1D list of $d_z$ values and as a point in a $d_z$-dimensional latent space. The latent-to-memory projection does not change this meaning; it reorganizes the same global information into multiple decoder-addressable memory slots.

Concretely, the decoder state and latent memory have different sequence axes:

$$
H \in \mathbb{R}^{B \times T \times d}, \quad M \in \mathbb{R}^{B \times N_m \times d}.
$$

In cross-attention, $T$ (decoder time steps) and $N_m$ (memory slots) are allowed to differ. Only the feature dimension $d$ must align for attention projections.

Each memory slot is best interpreted as a learned latent information container (for example, aspects of groove, density, accent tendencies, or timing style), not as a fixed symbolic time index. Therefore, slot $k$ is generally not guaranteed to correspond to a specific sixteenth-note position such as step index 3. Temporal structure can still be encoded implicitly across slots, but the slot-to-time mapping is learned and typically distributed.

An intuitive picture is: the latent code is a compact global summary, the projection expands this summary into several short latent notes, and cross-attention lets each decoder step retrieve whichever notes are most useful at that moment.

### Decoder with Cross-Attention Memory
The decoder is autoregressive. Given shifted target tokens, token and beat embeddings are fused and positionally encoded to produce decoder states:

$$
H \in \mathbb{R}^{B \times T \times d}.
$$

Cross-attention uses $H$ as queries and latent memory $M$ as keys and values:

$$
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V,
\quad Q = H, \quad K = V = M.
$$

This allows each decoding step to retrieve different latent information from memory slots, yielding token-dependent latent conditioning rather than a static global injection.

### Beat-Conditioned Timing Stream
To preserve rhythmic coherence, we maintain beat-progress tokens in both encoder and decoder paths. Beat indices are derived from step events and shifted appropriately for teacher forcing in training. This gives the model explicit intra-bar progression signals while retaining symbolic event flexibility.

### Training Objective
We optimize a VAE objective with token-level reconstruction loss and KL regularization:

$$
\mathcal{L} = \mathcal{L}_{\mathrm{rec}} + \beta \mathcal{L}_{\mathrm{KL}}.
$$

Reconstruction is cross-entropy over non-padding tokens. KL is computed per latent dimension and then normalized per valid token count:

$$
\mathcal{L}_{\mathrm{KL}}
= \frac{1}{B} \sum_{i=1}^{B}
\frac{\sum_{j=1}^{d_z} \max\!\left(\mathrm{KL}_{ij}, \lambda_{\mathrm{fb}}\right)}{\max(1, n_i)},
$$

where $\lambda_{\mathrm{fb}}$ is the free-bits threshold and $n_i$ is the number of valid tokens in sample $i$.

### KL Scheduling and Stabilization
We use annealed $\beta$ scheduling over training progress with piecewise smooth interpolation between user-defined anchor points. Early training emphasizes reconstruction, then gradually increases latent pressure. We apply gradient clipping for stability and mixed precision where supported.

### Inference and Interpolation
At inference, decoding can be deterministic (argmax) or sampled. For interpolation, we encode two sequences, use latent means, perform spherical interpolation in latent space, project each interpolated latent into memory slots, and decode each point autoregressively. This yields smooth rhythmic transitions while preserving global structural continuity.

### Why This Design
Compared with single-vector decoder conditioning, latent memory projection provides:

- Higher latent-access bandwidth via multiple memory slots.
- Step-wise selective retrieval through cross-attention.
- Better alignment with autoregressive decoding dynamics.
- Improved practical reconstruction and interpolation behavior in symbolic rhythm generation.

## Direct Comparison Template: Direct Latent-to-Memory vs Latent-to-Memory Expansion

### Comparison Goal
This section compares two conditioning strategies under matched training and evaluation settings:

- Baseline (Direct Latent-to-Memory): a single latent representation is passed directly to the decoder conditioning path.
- Proposed (Latent-to-Memory Expansion): the latent representation is projected into multiple learned memory slots and queried via cross-attention.

### Controlled Setup
Use identical settings for both models except the conditioning mechanism.

1. Data and split:
	- Dataset: [fill]
	- Train/validation/test split: [fill]
2. Model capacity controls:
	- Shared encoder and decoder backbone: [yes or no]
	- Parameter matching strategy: [equal parameters or report both]
3. Training controls:
	- Epochs: [fill]
	- Batch size: [fill]
	- Learning rate schedule: [fill]
	- KL schedule and free-bits settings: [fill]
	- Number of seeds: [fill]

### Evaluation Metrics
Report mean and standard deviation across seeds.

1. Reconstruction metrics:
	- Validation reconstruction loss: [fill]
	- Token-level perplexity: [fill]
2. Latent usage metrics:
	- KL per token: [fill]
	- Active latent dimensions (optional): [fill]
3. Generation quality metrics:
	- Interpolation smoothness score (define protocol): [fill]
	- Musical feature consistency (density, syncopation, timing deviation): [fill]
4. Subjective evaluation (optional):
	- Listening study protocol: [fill]
	- Preference rate: [fill]

### Quantitative Results Table (Template)

| Method | Params | Val Rec Loss (mean +/- std) | Val PPL (mean +/- std) | KL/Token (mean +/- std) | Interp Score (mean +/- std) |
|---|---:|---:|---:|---:|---:|
| Direct Latent-to-Memory | [fill] | [fill] | [fill] | [fill] | [fill] |
| Latent-to-Memory Expansion | [fill] | [fill] | [fill] | [fill] | [fill] |

### Qualitative Comparison (Template)
Provide paired examples using the same source sequences and interpolation endpoints.

1. Reconstruction pair examples:
	- Example IDs: [fill]
	- Notes: [fill]
2. Interpolation trajectory examples:
	- Endpoint pair IDs: [fill]
	- Notes on coherence and transitions: [fill]

### Interpretation Checklist
Use this checklist when interpreting differences.

1. Does expansion improve reconstruction without collapsing KL usage?
2. Are interpolation trajectories smoother and musically more coherent?
3. Are gains robust across seeds and not only in single runs?
4. Are improvements still present after controlling for parameter count?

### Reporting Summary (Fill-In)
In a direct comparison against the direct latent-to-memory baseline, latent-to-memory expansion [improved or did not improve] reconstruction ([fill]), perplexity ([fill]), and interpolation quality ([fill]). The KL usage pattern indicates [fill], suggesting [fill]. Overall, the expansion strategy [supports or does not support] the claim that sequence-structured latent memory improves decoder conditioning in this setting.

## References
1. Watters, N., Matthey, L., Burgess, C. P., and Lerchner, A. (2019). Spatial Broadcast Decoder: A Simple Architecture for Learning Disentangled Representations in VAEs. arXiv:1901.07017. https://doi.org/10.48550/arXiv.1901.07017
2. Radford, A., Metz, L., and Chintala, S. (2015). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. arXiv:1511.06434. https://doi.org/10.48550/arXiv.1511.06434
3. Roberts, A., Engel, J., Raffel, C., Hawthorne, C., and Eck, D. (2018). A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music. arXiv:1803.05428. https://doi.org/10.48550/arXiv.1803.05428
4. Liang, X., Wu, J., and Yin, Y. (2019). MIDI-Sandwich: Multi-model Multi-task Hierarchical Conditional VAE-GAN networks for Symbolic Single-track Music Generation. arXiv:1907.01607. https://doi.org/10.48550/arXiv.1907.01607
5. Liang, X., Wu, J., and Cao, J. (2019). MIDI-Sandwich2: RNN-based Hierarchical Multi-modal Fusion Generation VAE networks for multi-track symbolic music generation. arXiv:1909.03522. https://doi.org/10.48550/arXiv.1909.03522