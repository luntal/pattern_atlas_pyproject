"""ONNX export of a trained checkpoint (the logic of convert_to_onnx.py, usable from the server).

Like convert_to_onnx.py this exports the encoder and the decoder as separate graphs:
  - Encoder.forward  -> <model>_encoder.onnx   inputs: input_tokens [B, T, 4]            outputs: mu, logvar
  - Decoder.forward  -> <model>_decoder.onnx   inputs: tgt_tokens [B, T, 4], latent_z    outputs: logits
The decoder graph is the teacher forcing path; sample() / generate() are not part of the export.
The compound format has no beat tokens any more, so (unlike the triplet version) there is no 'beat_tokens' input.

In addition (always) the KV-cache step decoder is exported, two graphs for autoregressive generation:
  - <model>_decoder_init.onnx  inputs: latent_z [B, latent_dim]
                               outputs: mem_k_<i>, mem_v_<i> per layer  [B, H, n_memory_tokens, dh]  (cross-attention K/V)
  - <model>_decoder_step.onnx  inputs: frame [B, 4] (int64), pos [1] (int64, index of this frame, SOS = 0),
                                       past_k_<i>, past_v_<i> per layer [B, H, T, dh]  (T = 0 for the SOS frame),
                                       mem_k_<i>, mem_v_<i> per layer (from the init graph, constant during decoding)
                               outputs: logits [B, sum(field_vocab_sizes)], new_k_<i>, new_v_<i> per layer [B, H, T+1, dh]
The frame choice (Decoder.pick_frame with its constraints, sampling, EOS stop) is not part of the graphs; onnx_generate()
is a NumPy reference loop for it. Feed new_k/new_v back as past_k/past_v in the next step.

Output folder: models/<branch>/onnx/
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.branch_paths import project_root, safe_branch_name, get_current_branch
from modules.model import TransformerVAE
from modules.utilities import check_checkpoint_format

OPSET_VERSION = 14
_MODEL_FILE_RE = re.compile(r"^model_[A-Za-z0-9_.\-]+\.pt$")


def onnx_dir(branch: str | None = None) -> Path:
    return project_root() / "models" / safe_branch_name(branch if branch is not None else get_current_branch()) / "onnx"


def resolve_checkpoint(file_name: str, branch: str | None = None) -> Path:
    """Checkpoint file inside models/<branch>/ (plain file names only)."""
    name = str(file_name or "")
    if Path(name).name != name or not _MODEL_FILE_RE.match(name):
        raise ValueError(f"Invalid checkpoint file name '{name}'.")
    path = project_root() / "models" / safe_branch_name(branch if branch is not None else get_current_branch()) / name
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


class DecoderInitWrapper(nn.Module):
    """latent_z -> cross-attention K/V of the latent memory per layer (flat: k0, v0, k1, v1, ...)."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, z):
        dec = self.decoder
        D = dec.emb_size
        memory = dec.z_to_mem(z).reshape(z.size(0), dec.n_memory_tokens, D)
        outs = []
        for layer in dec.transformer_decoder.layers:
            mha = layer.multihead_attn
            w, b = mha.in_proj_weight, mha.in_proj_bias
            k = F.linear(memory, w[D:2 * D], b[D:2 * D])
            v = F.linear(memory, w[2 * D:], b[2 * D:])
            outs.append(dec._split_heads(k, mha.num_heads))
            outs.append(dec._split_heads(v, mha.num_heads))
        return tuple(outs)


class DecoderStepWrapper(nn.Module):
    """One decoding step with the KV cache as tensors (mirrors Decoder.decode_step, post-norm and pre-norm).

    forward(frame, pos, *cache) with cache = past_k0, past_v0, ..., mem_k0, mem_v0, ...
    returns (logits, new_k0, new_v0, new_k1, new_v1, ...).
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.n_layers = len(decoder.transformer_decoder.layers)

    def forward(self, frame, pos, *cache):
        dec = self.decoder
        n = self.n_layers
        D = dec.emb_size
        past, mem = cache[:2 * n], cache[2 * n:4 * n]

        x = (dec.token_embedding(frame) + dec.pos_embedding(pos)).unsqueeze(1)  # [B, 1, D]
        new_kv = []
        for i, layer in enumerate(dec.transformer_decoder.layers):
            sa, ca = layer.self_attn, layer.multihead_attn
            H = sa.num_heads

            def self_attn_block(h):
                q, k, v = F.linear(h, sa.in_proj_weight, sa.in_proj_bias).chunk(3, dim=-1)
                q, k, v = (dec._split_heads(t, H) for t in (q, k, v))
                k = torch.cat((past[2 * i], k), dim=2)
                v = torch.cat((past[2 * i + 1], v), dim=2)
                new_kv.append(k)
                new_kv.append(v)
                o = F.scaled_dot_product_attention(q, k, v)
                return sa.out_proj(o.transpose(1, 2).reshape(h.size(0), 1, D))

            def cross_attn_block(h):
                q = F.linear(h, ca.in_proj_weight[:D], ca.in_proj_bias[:D])
                o = F.scaled_dot_product_attention(dec._split_heads(q, H), mem[2 * i], mem[2 * i + 1])
                return ca.out_proj(o.transpose(1, 2).reshape(h.size(0), 1, D))

            def ff_block(h):
                return layer.linear2(layer.activation(layer.linear1(h)))

            if layer.norm_first:
                x = x + self_attn_block(layer.norm1(x))
                x = x + cross_attn_block(layer.norm2(x))
                x = x + ff_block(layer.norm3(x))
            else:
                x = layer.norm1(x + self_attn_block(x))
                x = layer.norm2(x + cross_attn_block(x))
                x = layer.norm3(x + ff_block(x))

        if dec.transformer_decoder.norm is not None:
            x = dec.transformer_decoder.norm(x)
        return (dec.fc_out(x[:, 0, :]),) + tuple(new_kv)


def onnx_generate(init_session, step_session, z, hparams, max_len=None, pad_after_eos: bool = True) -> np.ndarray:
    """Reference greedy generation with the exported init / step graphs (onnxruntime + NumPy only).

    Equivalent to Decoder.generate(z, sample=False, constrained=True). z: [B, latent_dim] float32.
    Returns frames [B, L, 4] (int64, without SOS), PAD frames after EOS.
    """
    z = np.asarray(z, dtype=np.float32)
    B = z.shape[0]
    n_layers = int(hparams.n_layers)
    H = int(hparams.n_heads)
    dh = int(hparams.d_model) // H
    sizes = [int(s) for s in hparams.field_vocab_sizes]
    starts = np.concatenate(([0], np.cumsum(sizes)))
    pad, sos, eos = int(hparams.pad_token), int(hparams.sos_token), int(hparams.eos_token)
    n_special = int(getattr(hparams, 'n_specials', 3))
    max_len = int(hparams.max_seq_len if max_len is None else min(max_len, hparams.max_seq_len))

    mem = init_session.run(None, {'latent_z': z})  # k0, v0, k1, v1, ...
    mem_feed = {}
    past = {}
    for i in range(n_layers):
        mem_feed[f'mem_k_{i}'], mem_feed[f'mem_v_{i}'] = mem[2 * i], mem[2 * i + 1]
        past[f'past_k_{i}'] = np.zeros((B, H, 0, dh), dtype=np.float32)
        past[f'past_v_{i}'] = np.zeros((B, H, 0, dh), dtype=np.float32)

    def masked_argmax(logits, invalid):
        return np.where(invalid, -np.inf, logits).argmax(axis=1)

    cur = np.full((B, 4), sos, dtype=np.int64)
    finished = np.zeros(B, dtype=bool)
    out = []
    for pos in range(max_len):
        if finished.all():
            break
        res = step_session.run(None, {'frame': cur, 'pos': np.array([pos], dtype=np.int64), **past, **mem_feed})
        logits = res[0]
        for i in range(n_layers):
            past[f'past_k_{i}'], past[f'past_v_{i}'] = res[1 + 2 * i], res[2 + 2 * i]
        step_l, pitch_l, vel_l, dev_l = (logits[:, starts[f]:starts[f + 1]] for f in range(4))

        # Decoder.pick_frame (greedy, constrained): step -> pitch -> velocity / deviation
        step_idx = np.arange(step_l.shape[1])[None, :]
        pitch_idx = np.arange(pitch_l.shape[1])[None, :]
        step_invalid = np.broadcast_to((step_idx < n_special) & (step_idx != eos), step_l.shape)
        prev_step, prev_pitch = cur[:, 0:1], cur[:, 1:2]
        is_note = (prev_step >= n_special) & (prev_pitch >= n_special)  # [B, 1]
        no_pitch_left = is_note & (prev_pitch >= pitch_l.shape[1] - 1)
        falls = is_note & (step_idx < prev_step) & (step_idx != eos)
        same_blocked = no_pitch_left & (step_idx == prev_step)
        step = masked_argmax(step_l, step_invalid | falls | same_blocked)
        same_step = is_note[:, 0] & (step == cur[:, 0])
        not_rising = same_step[:, None] & (pitch_idx <= prev_pitch)
        pitch = masked_argmax(pitch_l, np.broadcast_to(pitch_idx < n_special, pitch_l.shape) | not_rising)
        vel = masked_argmax(vel_l, np.broadcast_to(np.arange(vel_l.shape[1])[None, :] < n_special, vel_l.shape))
        dev = masked_argmax(dev_l, np.broadcast_to(np.arange(dev_l.shape[1])[None, :] < n_special, dev_l.shape))
        nxt = np.stack((step, pitch, vel, dev), axis=1).astype(np.int64)
        nxt[step == eos] = eos
        if pad_after_eos:
            nxt[finished] = pad
        finished = finished | (nxt[:, 0] == eos)
        out.append(nxt)
        cur = nxt
    if not out:
        return np.zeros((B, 0, 4), dtype=np.int64)
    return np.stack(out, axis=1)


def export_checkpoint_to_onnx(checkpoint_path, out_dir=None, log=print, opset_version: int = OPSET_VERSION) -> dict:
    """Export encoder, decoder and the KV-cache step decoder (init + step) of a checkpoint to ONNX.

    log(text) receives progress lines.
    Returns {encoder, decoder, decoder_init, decoder_step, out_dir, parameters}.
    Raises on failure (the caller reports the message).
    """
    checkpoint_path = Path(checkpoint_path)
    out_dir = Path(out_dir) if out_dir is not None else checkpoint_path.parent / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = out_dir / f"{checkpoint_path.stem}_encoder.onnx"
    decoder_path = out_dir / f"{checkpoint_path.stem}_decoder.onnx"
    init_path = out_dir / f"{checkpoint_path.stem}_decoder_init.onnx"
    step_path = out_dir / f"{checkpoint_path.stem}_decoder_step.onnx"

    log(f"Loading model: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    hparams_dict = checkpoint["hparams"]
    check_checkpoint_format(hparams_dict, checkpoint_path)
    hparams = SimpleNamespace(**hparams_dict)

    model = TransformerVAE(hparams)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    del checkpoint

    log("--" * 10)
    log("model config:")
    log(f"model dimensions: {hparams.d_model}")
    log(f"n_layers: {hparams.n_layers}")
    log(f"n_heads: {hparams.n_heads}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Total parameters in model: {total_params / 1e6:.2f}M")
    log("--" * 10)

    max_len = int(hparams.max_seq_len)
    n_fields = len(list(hparams.field_vocab_sizes))

    with torch.no_grad():
        # 2. Export Encoder (only calls Encoder.forward)
        log(f"Exporting encoder (opset {opset_version}) -> {encoder_path.name}")
        dummy_enc_input = torch.zeros((1, max_len, n_fields), dtype=torch.long)
        torch.onnx.export(
            model.encoder,
            (dummy_enc_input,),
            str(encoder_path),
            input_names=["input_tokens"],
            output_names=["mu", "logvar"],
            dynamic_axes={"input_tokens": {0: "batch_size", 1: "seq_len"}},
            opset_version=opset_version,
            dynamo=False,
        )
        log(f"Encoder written ({encoder_path.stat().st_size / 1e6:.2f} MB)")

        # 3. Export Decoder (only calls Decoder.forward; teacher forcing path)
        log(f"Exporting decoder (opset {opset_version}) -> {decoder_path.name}")
        dummy_dec_tokens = torch.zeros((1, max_len, n_fields), dtype=torch.long)
        dummy_z = torch.randn(1, int(hparams.latent_dim))
        torch.onnx.export(
            model.decoder,
            (dummy_dec_tokens, dummy_z),
            str(decoder_path),
            input_names=["tgt_tokens", "latent_z"],
            output_names=["logits"],
            dynamic_axes={
                "tgt_tokens": {0: "batch_size", 1: "seq_len"},
                "latent_z": {0: "batch_size"},
            },
            opset_version=opset_version,
            dynamo=False,
        )
        log(f"Decoder written ({decoder_path.stat().st_size / 1e6:.2f} MB)")

        # 4. KV-cache step decoder: init graph (latent -> cross-attention K/V) and step graph (one frame + cache)
        dec = model.decoder
        n_layers = len(dec.transformer_decoder.layers)
        H = int(hparams.n_heads)
        dh = int(hparams.d_model) // H
        n_mem = int(dec.n_memory_tokens)
        mem_names = [f"mem_{kv}_{i}" for i in range(n_layers) for kv in ("k", "v")]
        past_names = [f"past_{kv}_{i}" for i in range(n_layers) for kv in ("k", "v")]
        new_names = [f"new_{kv}_{i}" for i in range(n_layers) for kv in ("k", "v")]

        log(f"Exporting KV-cache decoder init (opset {opset_version}) -> {init_path.name}")
        torch.onnx.export(
            DecoderInitWrapper(dec).eval(),
            (dummy_z,),
            str(init_path),
            input_names=["latent_z"],
            output_names=mem_names,
            dynamic_axes={"latent_z": {0: "batch_size"}, **{n: {0: "batch_size"} for n in mem_names}},
            opset_version=opset_version,
            dynamo=False,
        )
        log(f"Decoder init written ({init_path.stat().st_size / 1e6:.2f} MB)")

        log(f"Exporting KV-cache decoder step (opset {opset_version}) -> {step_path.name}")
        past_len = 3  # dummy cache length (> 1 so that the length stays dynamic)
        dummy_frame = torch.full((1, n_fields), 3, dtype=torch.long)
        dummy_pos = torch.tensor([past_len], dtype=torch.long)
        dummy_past = [torch.randn(1, H, past_len, dh) for _ in past_names]
        dummy_mem = [torch.randn(1, H, n_mem, dh) for _ in mem_names]
        dynamic = {"frame": {0: "batch_size"}, "logits": {0: "batch_size"}}
        for n in past_names:
            dynamic[n] = {0: "batch_size", 2: "past_len"}
        for n in mem_names:
            dynamic[n] = {0: "batch_size"}
        for n in new_names:
            dynamic[n] = {0: "batch_size", 2: "new_len"}
        torch.onnx.export(
            DecoderStepWrapper(dec).eval(),
            (dummy_frame, dummy_pos, *dummy_past, *dummy_mem),
            str(step_path),
            input_names=["frame", "pos"] + past_names + mem_names,
            output_names=["logits"] + new_names,
            dynamic_axes=dynamic,
            opset_version=opset_version,
            dynamo=False,
        )
        log(f"Decoder step written ({step_path.stat().st_size / 1e6:.2f} MB, {n_layers} layers, "
            f"{len(past_names) + len(mem_names)} cache inputs)")

    log("Modules exported. sample() methods were naturally excluded.")
    return {"encoder": str(encoder_path), "decoder": str(decoder_path), "decoder_init": str(init_path),
            "decoder_step": str(step_path), "out_dir": str(out_dir), "parameters": int(total_params)}
