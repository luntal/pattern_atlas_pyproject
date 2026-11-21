"""Checks the compound TransformerVAE: field embedding, KV-cache vs. forward, constrained decoding, loss, overfit.

Run from the project root:  python tests/test_compound_model.py
"""
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from modules import compound  # noqa: E402
from modules.model import Decoder, FieldEmbedding, TransformerVAE, vae_loss  # noqa: E402

EOS, PAD, SOS = 2, 0, 1


def tiny_hparams(max_seq_len=48):
    return SimpleNamespace(
        d_model=32, n_heads=4, ff_dim=64, n_layers=2, latent_dim=16, n_memory_tokens=4, max_seq_len=max_seq_len,
        pad_token=PAD, sos_token=SOS, eos_token=EOS, width=32, height=54, n_pitch=54, n_vel=32, n_timing=33,
        field_vocab_sizes=[35, 57, 35, 36], token_format='compound_v1',
    )


def random_frames(rng, hp, batch, max_notes):
    seqs = []
    for _ in range(batch):
        n = rng.randrange(0, max_notes)
        notes = [(rng.randrange(32), rng.randrange(54), rng.randrange(32), rng.randrange(33)) for _ in range(n)]
        seqs.append(compound.notes_to_frames(notes, hp, max_len=hp.max_seq_len))
    return torch.from_numpy(np.stack(seqs).astype(np.int64))


def check_field_embedding():
    emb = FieldEmbedding([35, 57, 35, 36], 16)
    frames = torch.tensor([[[3, 4, 5, 6], [0, 0, 0, 0]]])
    out = emb(frames)
    assert out.shape == (1, 2, 16)
    assert torch.allclose(out[0, 1], torch.zeros(16)), "PAD frame embeds to zero"
    manual = sum(e(frames[..., i]) for i, e in enumerate(emb.embeddings)) / 2.0
    assert torch.allclose(out, manual)


def check_cache_matches_forward(norm_first):
    hp = tiny_hparams()
    torch.manual_seed(0)
    dec = Decoder(hp)
    if norm_first:
        for layer in dec.transformer_decoder.layers:
            layer.norm_first = True
    dec.eval()
    rng = random.Random(1)
    frames = random_frames(rng, hp, 5, 30)
    dec_in = torch.cat([torch.full((5, 1, 4), SOS), frames[:, :-1]], dim=1)
    dec_in = dec_in.masked_fill((dec_in[..., 0] == EOS).unsqueeze(-1), PAD)
    dec_in[:, 7] = PAD  # exercise the padding handling in the middle
    z = torch.randn(5, hp.latent_dim)
    with torch.no_grad():
        full = dec(dec_in, z)
        assert full.shape == (5, hp.max_seq_len, sum(hp.field_vocab_sizes))
        cache = dec.init_cache(z)
        for t in range(dec_in.size(1)):
            step = dec.decode_step(dec_in[:, t], cache)
            assert torch.allclose(step, full[:, t], atol=1e-4), f"logits differ at t={t} (norm_first={norm_first})"


@torch.no_grad()
def legacy_greedy(dec, z, max_len):
    """Reference: re-run the whole sequence for every new frame (no cache), same pick_frame."""
    B = z.size(0)
    cur = torch.full((B, 1, 4), SOS, dtype=torch.long)
    finished = torch.zeros(B, dtype=torch.bool)
    out = []
    for _ in range(max_len):
        if cur.size(1) > dec.max_seq_len or finished.all():
            break
        logits = dec(cur, z)[:, -1]
        nxt = dec.pick_frame(logits, cur[:, -1])
        nxt = torch.where(finished.unsqueeze(1), torch.zeros_like(nxt), nxt)
        finished |= nxt[:, 0] == EOS
        out.append(nxt)
        cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
    return torch.stack(out, dim=1)


def check_generate():
    hp = tiny_hparams()
    torch.manual_seed(3)
    dec = Decoder(hp).eval()
    z = torch.randn(6, hp.latent_dim) * 3
    got = dec.generate(z, max_len=hp.max_seq_len)
    assert got.dim() == 3 and got.size(-1) == 4
    ref = legacy_greedy(dec, z, hp.max_seq_len)
    assert torch.equal(got, ref), "cached generate differs from the uncached reference"
    for i in range(z.size(0)):
        single = dec.generate(z[i:i + 1], max_len=hp.max_seq_len)[0]
        eos = (single[:, 0] == EOS).nonzero()
        n = int(eos[0]) + 1 if len(eos) else single.size(0)
        assert torch.equal(single[:n], got[i][:n]), f"batch vs single differ for item {i}"
        assert bool((got[i][n:] == PAD).all()), f"frames after EOS are not PAD for item {i}"
        check_constraints(got[i])
    # sampling with constraints keeps the order as well
    torch.manual_seed(5)
    sampled = dec.generate(z, max_len=hp.max_seq_len, sample=True, temperature=1.5)
    for row in sampled:
        check_constraints(row)
    assert torch.equal(dec.sample(z, hp.max_seq_len, sample=False), got)


def check_constraints(frames):
    prev = None
    for frame in frames.tolist():
        if frame[0] in (EOS, PAD):
            assert frame == [frame[0]] * 4, f"special frame must carry the id in all fields: {frame}"
            break
        assert all(v >= 3 for v in frame), f"special id inside a note frame: {frame}"
        if prev is not None:
            assert frame[0] >= prev[0], f"step fell: {prev} -> {frame}"
            if frame[0] == prev[0]:
                assert frame[1] > prev[1], f"pitch not rising at the same step: {prev} -> {frame}"
        prev = frame


def check_pick_frame_edges():
    hp = tiny_hparams()
    dec = Decoder(hp).eval()
    sizes = hp.field_vocab_sizes
    total = sum(sizes)

    def logits_with(step_pref, pitch_pref):
        lg = torch.zeros(1, total)
        s0, p0 = 0, sizes[0]
        lg[0, s0 + step_pref] = 10.0
        lg[0, p0 + pitch_pref] = 10.0
        return lg

    # preferred step falls below the previous one -> constrained picks a later step (or EOS)
    prev = torch.tensor([[3 + 20, 3 + 10, 10, 10]])
    out = dec.pick_frame(logits_with(3 + 5, 3 + 1), prev)
    assert out[0, 0] >= prev[0, 0] or out[0, 0] == EOS, out
    unconstrained = dec.pick_frame(logits_with(3 + 5, 3 + 1), prev, constrained=False)
    assert unconstrained[0, 0] == 3 + 5, unconstrained
    # same step: the pitch must rise
    out = dec.pick_frame(logits_with(3 + 20, 3 + 4), prev)
    assert out[0, 0] == 3 + 20 and out[0, 1] > prev[0, 1], out
    # pitch 53 at step 20: no pitch left -> next step
    prev = torch.tensor([[3 + 20, 3 + 53, 10, 10]])
    out = dec.pick_frame(logits_with(3 + 20, 3 + 53), prev)
    assert out[0, 0] > 3 + 20 or out[0, 0] == EOS, out
    # pitch 53 at step 31: only EOS is left
    prev = torch.tensor([[3 + 31, 3 + 53, 10, 10]])
    out = dec.pick_frame(logits_with(3 + 31, 3 + 53), prev)
    assert out[0].tolist() == [EOS] * 4, out
    # specials are never chosen inside a note, EOS gives an EOS frame
    lg = torch.zeros(1, total)
    lg[0, sizes[0] + PAD] = 50.0  # pitch head prefers PAD
    lg[0, 3] = 5.0
    out = dec.pick_frame(lg, torch.full((1, 4), SOS))
    assert out[0, 0] == 3 and out[0, 1] >= 3, out
    lg = torch.zeros(1, total)
    lg[0, EOS] = 5.0
    assert dec.pick_frame(lg, torch.full((1, 4), SOS))[0].tolist() == [EOS] * 4


def check_loss_masks():
    hp = tiny_hparams(max_seq_len=6)
    dec = Decoder(hp)
    slices = dec.field_slices
    targets = torch.tensor([[[3, 4, 5, 6], [4, 5, 6, 7], [EOS] * 4, [PAD] * 4, [PAD] * 4, [PAD] * 4]])
    torch.manual_seed(0)
    logits = torch.randn(1, 6, sum(hp.field_vocab_sizes))
    mu, logvar = torch.zeros(1, hp.latent_dim), torch.zeros(1, hp.latent_dim)
    total, recon, kl, ppl, n_frames, field_ce = vae_loss(logits, targets, mu, logvar, beta=0.0, field_slices=slices)
    assert field_ce.shape == (4,) and int(n_frames) == 3
    # changing non-step logits at the EOS frame or anything at PAD frames does not change the loss
    changed = logits.clone()
    changed[0, 2, slices[1][0]:] += torch.randn(sum(hp.field_vocab_sizes) - slices[1][0]) * 5
    changed[0, 3:] += 7.0 * torch.randn_like(changed[0, 3:])
    total2 = vae_loss(changed, targets, mu, logvar, beta=0.0, field_slices=slices)[0]
    assert torch.allclose(total, total2), (total, total2)
    # but the step head at the EOS frame counts
    changed = logits.clone()
    changed[0, 2, EOS] += 10.0
    assert vae_loss(changed, targets, mu, logvar, beta=0.0, field_slices=slices)[0] < total
    # recon = sum of field CE sums / frames
    manual = 0.0
    for f, (s, e) in enumerate(slices):
        tgt = targets[0, :, f].clone()
        if f > 0:
            tgt[2] = PAD
        manual += torch.nn.functional.cross_entropy(logits[0, :, s:e], tgt, ignore_index=PAD, reduction='sum') / 3
    assert torch.allclose(recon, manual), (recon, manual)
    # KL per frame
    mu = torch.ones(1, hp.latent_dim)
    kl = vae_loss(logits, targets, mu, logvar, beta=1.0, field_slices=slices)[2]
    assert torch.allclose(kl, torch.tensor(0.5 * hp.latent_dim / 3)), kl
    # KL parts: pre beta (kl_raw) and the true KL without the free-bits floor (kl_raw_nofb)
    out = vae_loss(logits, targets, mu, logvar, beta=2.0, free_bits=1.0, field_slices=slices, return_kl_parts=True)
    assert len(out) == 7 and out[6].shape == (2,)
    assert torch.allclose(out[2], torch.tensor(2.0 * 1.0 * hp.latent_dim / 3)), out[2]  # post beta, with free bits
    assert torch.allclose(out[6][0], torch.tensor(1.0 * hp.latent_dim / 3)), out[6]  # pre beta, with free bits
    assert torch.allclose(out[6][1], torch.tensor(0.5 * hp.latent_dim / 3)), out[6]  # pre beta, true KL
    assert len(vae_loss(logits, targets, mu, logvar, beta=1.0, field_slices=slices)) == 6
    # field weights
    w_total = vae_loss(logits, targets, torch.zeros(1, hp.latent_dim), logvar, beta=0.0, field_slices=slices,
                       field_weights=[1, 0, 0, 0])[1]
    step_only = torch.nn.functional.cross_entropy(logits[0, :, :35], targets[0, :, 0], ignore_index=PAD, reduction='sum') / 3
    assert torch.allclose(w_total, step_only)


def check_checkpoint_format():
    from modules.utilities import LegacyCheckpointError, check_checkpoint_format as check
    check({'token_format': 'compound_v1'}, 'model_x.pt')
    for bad in ({}, {'vocab_size': 123}, {'token_format': 'triplet'}):
        try:
            check(bad, 'model_old.pt')
        except LegacyCheckpointError as exc:
            assert 'model_old.pt' in str(exc) and 'compound_v1' in str(exc)
            continue
        raise AssertionError(f'legacy checkpoint accepted: {bad}')


def check_overfit():
    hp = tiny_hparams(max_seq_len=24)
    hp.d_model, hp.ff_dim, hp.latent_dim = 64, 128, 32
    torch.manual_seed(0)
    rng = random.Random(0)
    frames = random_frames(rng, hp, 8, 20)
    model = TransformerVAE(hp)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    slices = model.decoder.field_slices
    model.train()
    first = None
    for it in range(400):
        logits, mu, logvar = model(frames)
        loss, recon, kl, ppl, n, field_ce = vae_loss(logits, frames, mu, logvar, beta=0.0, field_slices=slices)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = float(recon.detach())
    last = float(recon.detach())
    assert last < 0.1 * first, (first, last)
    model.eval()
    with torch.no_grad():
        mu, _ = model.encoder(frames)
        gen = model.decoder.generate(mu, max_len=hp.max_seq_len)
    exact = sum(int(np.array_equal(compound.frames_to_raw(gen[i].numpy(), hp), compound.frames_to_raw(frames[i].numpy(), hp)))
                for i in range(8))
    print(f"overfit: recon {first:.3f} -> {last:.4f}, exact greedy reconstructions {exact}/8")
    assert exact >= 6, exact


if __name__ == '__main__':
    check_field_embedding()
    check_cache_matches_forward(False)
    check_cache_matches_forward(True)
    check_generate()
    check_pick_frame_edges()
    check_loss_masks()
    check_checkpoint_format()
    check_overfit()
    print("ALL OK: compound model")
