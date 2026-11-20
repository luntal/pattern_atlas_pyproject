"""Checks that the KV-cache decoding of `Decoder` matches the full (uncached) forward pass.

Run from the project root:  python tests/test_kv_cache.py
"""
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from modules.model import Decoder  # noqa: E402


def make_decoder(norm_first=False):
    hp = SimpleNamespace(
        vocab_size=40, d_model=32, n_heads=4, ff_dim=64, n_layers=3, latent_dim=16,
        n_memory_tokens=4, total_steps=12, max_seq_len=48,
        pad_token=0, sos_token=1, eos_token=2, step_token=3,
    )
    torch.manual_seed(0)
    dec = Decoder(hp)
    if norm_first:
        for layer in dec.transformer_decoder.layers:
            layer.norm_first = True
    return dec.eval(), hp


def beat_from_tokens(tokens, step_token, max_beat_index):
    beat = torch.cumsum((tokens == step_token).long(), dim=1) + 1
    beat[:, 0] = 0
    return beat.clamp(max=max_beat_index)


@torch.no_grad()
def legacy_greedy(dec, z, max_len):
    """Original loop: re-run the whole sequence for every new token."""
    B = z.size(0)
    memory = dec.z_to_mem(z).view(B, dec.n_memory_tokens, dec.emb_size)
    max_beat = dec.beat_embedding.num_embeddings - 1
    cur = torch.full((B, 1), dec.sos_token, dtype=torch.long)
    for _ in range(max_len):
        T = cur.size(1)
        if T > dec.max_seq_len:
            break
        beat = beat_from_tokens(cur, dec.step_token, max_beat)
        emb = dec.projection(torch.cat((dec.token_embedding(cur), dec.beat_embedding(beat)), dim=-1))
        emb = emb + dec.pos_embedding(torch.arange(T)).unsqueeze(0)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(T)
        out = dec.transformer_decoder(tgt=emb, memory=memory, tgt_mask=mask,
                                      tgt_key_padding_mask=(cur == dec.pad_token).float())
        nxt = dec.fc_out(out)[:, -1].argmax(-1, keepdim=True)
        cur = torch.cat([cur, nxt], dim=1)
    return cur[:, 1:]


def check_logits(norm_first):
    dec, hp = make_decoder(norm_first)
    B, T = 5, 30
    z = torch.randn(B, hp.latent_dim)
    tokens = torch.randint(3, hp.vocab_size, (B, T))
    tokens[:, 0] = hp.sos_token
    tokens[torch.rand(B, T) < 0.2] = hp.step_token
    tokens[:, 10] = hp.pad_token  # exercise the padding handling
    tokens[:, 0] = hp.sos_token
    beat = beat_from_tokens(tokens, hp.step_token, hp.total_steps - 1)
    with torch.no_grad():
        full = dec(tokens, z, beat)
        cache = dec.init_cache(z)
        for t in range(T):
            step = dec.decode_step(tokens[:, t], cache)
            assert torch.allclose(step, full[:, t], atol=1e-4), f"logits differ at t={t} (norm_first={norm_first})"


def check_greedy():
    dec, hp = make_decoder()
    z = torch.randn(6, hp.latent_dim)
    max_len = 40
    ref = legacy_greedy(dec, z, max_len)
    got = dec.generate(z, max_len=max_len, pad_after_eos=False)
    assert torch.equal(ref, got), "greedy tokens differ"
    # Batched result must equal per-sample results (up to and including the first EOS).
    padded = dec.generate(z, max_len=max_len, pad_after_eos=True)
    for i in range(z.size(0)):
        single = dec.generate(z[i:i + 1], max_len=max_len, pad_after_eos=False)[0]
        eos_pos = (single == dec.eos_token).nonzero()
        n = int(eos_pos[0]) + 1 if len(eos_pos) else single.numel()
        assert torch.equal(single[:n], got[i][:n]), f"batch vs single differ for item {i}"
        assert torch.equal(single[:n], padded[i][:n]), f"padded batch vs single differ for item {i}"
        assert bool((padded[i][n:] == dec.pad_token).all()), f"tokens after EOS are not PAD for item {i}"
    # sample() keeps its legacy output shape.
    assert torch.equal(dec.sample(z, max_len, temperature=0.1, sample=False), ref)


if __name__ == '__main__':
    check_logits(False)
    check_logits(True)
    check_greedy()
    print("OK: KV-cache decoding matches the uncached decoder.")
