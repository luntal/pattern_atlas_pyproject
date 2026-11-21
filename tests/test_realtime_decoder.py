"""Checks the real-time decoder worker pool (rbf_server_app.RealtimeDecoderManager).

Run from the project root:  python tests/test_realtime_decoder.py
"""
import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from modules import compound  # noqa: E402
from modules.model import Decoder  # noqa: E402
import rbf_server_app as app  # noqa: E402

PAD, SOS, EOS = 0, 1, 2


def make_hparams():
    return SimpleNamespace(
        d_model=32, n_heads=4, ff_dim=64, n_layers=2, latent_dim=16, n_memory_tokens=4, max_seq_len=40,
        pad_token=PAD, sos_token=SOS, eos_token=EOS, field_vocab_sizes=[35, 57, 35, 36],
        width=32, n_timing=33, n_pitch=54, n_vel=32, height=54, norm_first=True, dropout=0.0,
    )


def make_decoder(hp):
    torch.manual_seed(0)
    return Decoder(hp).eval()


def wait_for(cond, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def reference_matrix(dec, hp, latent):
    z = torch.from_numpy(latent).unsqueeze(0)
    with torch.no_grad():
        gen = dec.generate(z, max_len=hp.max_seq_len, sample=False, constrained=True, pad_after_eos=True)
    row = gen[0].numpy().astype(np.int64)
    eos = np.flatnonzero(row[:, 0] == EOS)
    seq = row[: int(eos[0]) + 1] if eos.size else row
    return compound.frames_to_matrix(seq, hp)


def test_check_every_same_result():
    hp = make_hparams()
    dec = make_decoder(hp)
    z = torch.randn(6, hp.latent_dim)
    a = dec.generate(z, max_len=hp.max_seq_len, pad_after_eos=True, check_every=1)
    b = dec.generate(z, max_len=hp.max_seq_len, pad_after_eos=True, check_every=8)
    n = a.size(1)
    assert torch.equal(a, b[:, :n]), "content differs"
    assert (b[:, n:, 0] == PAD).all(), "extra frames must be PAD"
    print("check_every ok")


def test_result_matches_reference():
    hp = make_hparams()
    dec = make_decoder(hp)
    mgr = app.RealtimeDecoderManager(dec, hp, max_workers=2, stream_partial=False)
    rng = np.random.default_rng(1)
    latent = rng.normal(size=hp.latent_dim).astype(np.float32)
    mgr.request_decode(latent, np.zeros((hp.height, hp.width * hp.n_timing), dtype=np.float32))
    got = {}

    def poll():
        upd = mgr.consume_latest_update()
        if upd is not None and upd["status"] == "done":
            got.update(upd)
        return bool(got)

    assert wait_for(poll), "no result"
    assert np.allclose(got["matrix"], reference_matrix(dec, hp, latent))
    assert got["decode_ms"] > 0
    print("matches reference, decode_ms=%.1f" % got["decode_ms"])


def test_newest_wins_and_nothing_cancelled():
    """Many fast requests: running jobs finish, only the newest request waits, the last latent is what ends up shown."""
    hp = make_hparams()
    dec = make_decoder(hp)
    slow = threading.Semaphore(0)

    class SlowDecoder:
        """Delegates to the real decoder but makes generate slow, so requests pile up."""
        def __init__(self, inner):
            self.inner = inner
            self.max_seq_len = inner.max_seq_len
            self.calls = 0

        def parameters(self):
            return self.inner.parameters()

        def generate(self, *a, **kw):
            self.calls += 1
            time.sleep(0.15)
            return self.inner.generate(*a, **kw)

    sd = SlowDecoder(dec)
    mgr = app.RealtimeDecoderManager(sd, hp, max_workers=2, stream_partial=False)
    rng = np.random.default_rng(2)
    latents = [rng.normal(size=hp.latent_dim).astype(np.float32) for _ in range(10)]
    for lat in latents:
        mgr.request_decode(lat)
        time.sleep(0.01)
    expected = reference_matrix(dec, hp, latents[-1])
    shown = {}

    def poll():
        upd = mgr.consume_latest_update()
        if upd is not None:
            shown.update(upd)
        return bool(shown) and shown["status"] == "done" and np.allclose(shown["matrix"], expected) \
            and mgr._active_workers == 0

    assert wait_for(poll, 30), "the newest latent was never shown"
    # at most 2 running + the newest waiting request were ever started -> far fewer than 10 decodes
    assert sd.calls <= 4, f"too many decodes: {sd.calls}"
    assert sd.calls >= 2, "jobs were not run in parallel / completed"
    print(f"newest wins, {sd.calls} of 10 requests decoded")


def test_older_result_never_replaces_newer():
    hp = make_hparams()
    dec = make_decoder(hp)
    mgr = app.RealtimeDecoderManager(dec, hp, max_workers=2, stream_partial=False)
    m_new = np.ones((hp.height, hp.width * hp.n_timing), dtype=np.float32)
    m_old = np.zeros_like(m_new)
    assert mgr._publish_update(5, m_new, 1, "done", decode_ms=1.0)
    assert not mgr._publish_update(3, m_old, 1, "done", decode_ms=1.0)
    assert mgr.consume_latest_update()["matrix"].sum() == m_new.sum()
    print("older never replaces newer ok")


def test_discard_drops_running_results():
    hp = make_hparams()
    dec = make_decoder(hp)

    class SlowDecoder:
        def __init__(self, inner):
            self.inner = inner
            self.max_seq_len = inner.max_seq_len

        def parameters(self):
            return self.inner.parameters()

        def generate(self, *a, **kw):
            time.sleep(0.2)
            return self.inner.generate(*a, **kw)

    mgr = app.RealtimeDecoderManager(SlowDecoder(dec), hp, max_workers=2, stream_partial=False)
    mgr.consume_latest_update()  # the very first call always returns the initial idle state
    mgr.request_decode(np.zeros(hp.latent_dim, dtype=np.float32))
    time.sleep(0.05)
    mgr.discard()
    assert wait_for(lambda: mgr._active_workers == 0)
    assert mgr.consume_latest_update() is None, "a discarded job must not publish"
    mgr.request_decode(np.ones(hp.latent_dim, dtype=np.float32))
    assert wait_for(lambda: (mgr.consume_latest_update() or {}).get("status") == "done"), "decoding after discard"
    print("discard ok")


def test_streaming_mode_matches():
    hp = make_hparams()
    dec = make_decoder(hp)
    mgr = app.RealtimeDecoderManager(dec, hp, max_workers=1, stream_partial=True)
    latent = np.random.default_rng(3).normal(size=hp.latent_dim).astype(np.float32)
    mgr.request_decode(latent)
    got = {}

    def poll():
        upd = mgr.consume_latest_update()
        if upd is not None and upd["status"] == "done":
            got.update(upd)
        return bool(got)

    assert wait_for(poll)
    assert np.allclose(got["matrix"], reference_matrix(dec, hp, latent))
    print("streaming mode ok")


def test_shutdown():
    hp = make_hparams()
    dec = make_decoder(hp)
    mgr = app.RealtimeDecoderManager(dec, hp, max_workers=2, stream_partial=False)
    for _ in range(5):
        mgr.request_decode(np.random.randn(hp.latent_dim).astype(np.float32))
    mgr.shutdown(timeout=20)
    assert mgr._active_workers == 0
    mgr.request_decode(np.zeros(hp.latent_dim, dtype=np.float32))  # ignored after shutdown
    assert mgr._active_workers == 0
    print("shutdown ok")


if __name__ == '__main__':
    test_check_every_same_result()
    test_result_matches_reference()
    test_newest_wins_and_nothing_cancelled()
    test_older_result_never_replaces_newer()
    test_discard_drops_running_results()
    test_streaming_mode_matches()
    test_shutdown()
    print('ALL OK')
