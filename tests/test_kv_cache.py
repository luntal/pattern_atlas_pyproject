"""Checks that the KV-cache decoding of `Decoder` matches the full (uncached) forward pass (compound frames).

Run from the project root:  python tests/test_kv_cache.py
"""
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from modules.model import Decoder  # noqa: E402

PAD, SOS, EOS = 0, 1, 2


def make_decoder(norm_first=False):
    hp = SimpleNamespace(
        d_model=32, n_heads=4, ff_dim=64, n_layers=3, latent_dim=16, n_memory_tokens=4, max_seq_len=48,
        pad_token=PAD, sos_token=SOS, eos_token=EOS, field_vocab_sizes=[35, 57, 35, 36],
    )
    torch.manual_seed(0)
    dec = Decoder(hp)
    if norm_first:
        for layer in dec.transformer_decoder.layers:
            layer.norm_first = True
    return dec.eval(), hp


def random_frames(B, T, sizes):
    cols = [torch.randint(3, size, (B, T)) for size in sizes]
    return torch.stack(cols, dim=-1)


@torch.no_grad()
def legacy_greedy(dec, z, max_len):
    """Original loop: re-run the whole sequence for every new frame."""
    B = z.size(0)
    cur = torch.full((B, 1, 4), SOS, dtype=torch.long)
    for _ in range(max_len):
        if cur.size(1) > dec.max_seq_len:
            break
        nxt = dec.pick_frame(dec(cur, z)[:, -1], cur[:, -1])
        cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
    return cur[:, 1:]


def check_logits(norm_first):
    dec, hp = make_decoder(norm_first)
    B, T = 5, 30
    z = torch.randn(B, hp.latent_dim)
    frames = random_frames(B, T, hp.field_vocab_sizes)
    frames[:, 0] = SOS
    frames[:, 10] = PAD  # exercise the padding handling
    with torch.no_grad():
        full = dec(frames, z)
        cache = dec.init_cache(z)
        for t in range(T):
            step = dec.decode_step(frames[:, t], cache)
            assert torch.allclose(step, full[:, t], atol=1e-4), f"logits differ at t={t} (norm_first={norm_first})"


def check_greedy():
    dec, hp = make_decoder()
    z = torch.randn(6, hp.latent_dim)
    max_len = 40
    ref = legacy_greedy(dec, z, max_len)
    got = dec.generate(z, max_len=max_len, pad_after_eos=False)
    assert torch.equal(ref[:, :got.size(1)], got), "greedy frames differ"
    # Batched result must equal per-sample results (up to and including the first EOS).
    padded = dec.generate(z, max_len=max_len, pad_after_eos=True)
    for i in range(z.size(0)):
        single = dec.generate(z[i:i + 1], max_len=max_len, pad_after_eos=False)[0]
        eos_pos = (single[:, 0] == EOS).nonzero()
        n = int(eos_pos[0]) + 1 if len(eos_pos) else single.size(0)
        assert torch.equal(single[:n], padded[i][:n]), f"padded batch vs single differ for item {i}"
        assert bool((padded[i][n:] == PAD).all()), f"frames after EOS are not PAD for item {i}"


if __name__ == '__main__':
    check_logits(False)
    check_logits(True)
    check_greedy()
    print("OK: KV-cache decoding matches the uncached decoder.")
