"""ONNX export (Latent Field > Model Checkpoint > Export to ONNX): module, server events and numerical match.

Runs in a temporary project root, so nothing is written to the real models/ folder.
Prints 'ALL OK' on success.
"""
import glob
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import numpy as np
import torch

from modules import onnx_export
from modules.branch_paths import get_current_branch, safe_branch_name

BRANCH = safe_branch_name(get_current_branch())
sources = sorted(glob.glob(os.path.join(ROOT, 'models', BRANCH, 'model_best_val_*.pt')))
if not sources:
    print(f'SKIP: no checkpoint in models/{BRANCH}/ to export')
    sys.exit(0)
SOURCE = sources[-1]

tmp = Path(tempfile.mkdtemp(prefix='onnx_test_'))
try:
    (tmp / 'models' / BRANCH).mkdir(parents=True)
    ckpt_name = os.path.basename(SOURCE)
    # all checkpoints of this training, so that whichever one the server has loaded exists in the temporary root
    stamp = ckpt_name.split('model_best_val_')[1]
    for f in glob.glob(os.path.join(ROOT, 'models', BRANCH, f'model*{stamp}')):
        shutil.copy(f, tmp / 'models' / BRANCH / os.path.basename(f))
    onnx_export.project_root = lambda: tmp  # redirect models/<branch>/ and models/<branch>/onnx/

    # --- file name validation
    for bad in ['../x.pt', 'a/b.pt', 'model.onnx', '', 'model_x.pt']:
        try:
            onnx_export.resolve_checkpoint(bad)
        except (ValueError, FileNotFoundError):
            pass
        else:
            raise AssertionError(f'{bad!r} must be rejected')
    assert onnx_export.onnx_dir() == tmp / 'models' / BRANCH / 'onnx'
    print('ok name validation')

    # --- module: export + numerical match with the PyTorch model
    lines = []
    result = onnx_export.export_checkpoint_to_onnx(onnx_export.resolve_checkpoint(ckpt_name), onnx_export.onnx_dir(),
                                                  log=lines.append)
    stem = ckpt_name[:-3]
    enc_file = onnx_export.onnx_dir() / f'{stem}_encoder.onnx'
    dec_file = onnx_export.onnx_dir() / f'{stem}_decoder.onnx'
    assert enc_file.is_file() and dec_file.is_file(), result
    assert any('Total parameters' in l for l in lines) and any('exported' in l for l in lines), lines
    import onnxruntime as ort
    from types import SimpleNamespace
    from modules.model import TransformerVAE
    ck = torch.load(tmp / 'models' / BRANCH / ckpt_name, map_location='cpu')
    hp = SimpleNamespace(**ck['hparams'])
    model = TransformerVAE(hp)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    rng = np.random.default_rng(0)
    for batch, length in [(1, hp.max_seq_len), (3, 40)]:  # dynamic batch and sequence length
        frames = np.zeros((batch, length, 4), dtype=np.int64)
        n = min(length - 1, 12)
        frames[:, :n, 0] = np.sort(rng.integers(3, 35, size=(batch, n)), axis=1)
        frames[:, :n, 1] = rng.integers(3, 57, size=(batch, n))
        frames[:, :n, 2] = rng.integers(3, 35, size=(batch, n))
        frames[:, :n, 3] = rng.integers(3, 36, size=(batch, n))
        frames[:, n] = 2
        enc = ort.InferenceSession(str(enc_file))
        dec = ort.InferenceSession(str(dec_file))
        mu, logvar = enc.run(None, {'input_tokens': frames})
        logits = dec.run(None, {'tgt_tokens': frames, 'latent_z': mu})[0]
        with torch.no_grad():
            tmu, tlv = model.encoder(torch.from_numpy(frames))
            tlogits = model.decoder(torch.from_numpy(frames), tmu)
        assert np.abs(tmu.numpy() - mu).max() < 1e-4 and np.abs(tlv.numpy() - logvar).max() < 1e-4
        assert np.abs(tlogits.numpy() - logits).max() < 1e-3, np.abs(tlogits.numpy() - logits).max()
    print('ok export + onnxruntime matches torch (batch 1 / 3, length', hp.max_seq_len, '/ 40)')

    # --- KV-cache step decoder (always exported): init + step graphs
    init_file = onnx_export.onnx_dir() / f'{stem}_decoder_init.onnx'
    step_file = onnx_export.onnx_dir() / f'{stem}_decoder_step.onnx'
    assert init_file.is_file() and step_file.is_file(), os.listdir(onnx_export.onnx_dir())
    assert result['decoder_init'] == str(init_file) and result['decoder_step'] == str(step_file), result
    assert any('KV-cache decoder step' in l for l in lines), lines
    init_sess = ort.InferenceSession(str(init_file))
    step_sess = ort.InferenceSession(str(step_file))
    dec = model.decoder
    n_layers = len(dec.transformer_decoder.layers)
    H = hp.n_heads
    dh = hp.d_model // H
    for batch in (1, 3):
        z = rng.standard_normal((batch, hp.latent_dim)).astype(np.float32)
        cache = dec.init_cache(torch.from_numpy(z))
        mem = init_sess.run(None, {'latent_z': z})
        for i in range(n_layers):
            for j in range(2):
                assert np.abs(mem[2 * i + j] - cache['mem_kv'][i][j].numpy()).max() < 1e-4
        mem_feed = {}
        past = {}
        for i in range(n_layers):
            mem_feed[f'mem_k_{i}'], mem_feed[f'mem_v_{i}'] = mem[2 * i], mem[2 * i + 1]
            past[f'past_k_{i}'] = np.zeros((batch, H, 0, dh), dtype=np.float32)  # empty cache for the SOS frame
            past[f'past_v_{i}'] = np.zeros((batch, H, 0, dh), dtype=np.float32)
        cur = np.full((batch, 4), hp.sos_token, dtype=np.int64)
        for pos in range(10):
            res = step_sess.run(None, {'frame': cur, 'pos': np.array([pos], dtype=np.int64), **past, **mem_feed})
            with torch.no_grad():
                ref = dec.decode_step(torch.from_numpy(cur), cache)
            assert res[0].shape == tuple(ref.shape), (res[0].shape, ref.shape)
            assert np.abs(res[0] - ref.numpy()).max() < 1e-3, (pos, np.abs(res[0] - ref.numpy()).max())
            for i in range(n_layers):
                past[f'past_k_{i}'], past[f'past_v_{i}'] = res[1 + 2 * i], res[2 + 2 * i]
                assert past[f'past_k_{i}'].shape == (batch, H, pos + 1, dh)
            cur = np.stack([rng.integers(3, s, size=batch) for s in hp.field_vocab_sizes], axis=1).astype(np.int64)
    print('ok init / step graphs match init_cache / decode_step (batch 1 / 3, 10 steps)')

    z = rng.standard_normal((4, hp.latent_dim)).astype(np.float32)
    with torch.no_grad():
        z = np.concatenate([z, model.encoder(torch.from_numpy(frames))[0].numpy()[:1]], axis=0)  # + a real latent
        ref_frames = dec.generate(torch.from_numpy(z), check_every=1).numpy()
    onnx_frames = onnx_export.onnx_generate(init_sess, step_sess, z, hp)
    assert onnx_frames.shape == ref_frames.shape, (onnx_frames.shape, ref_frames.shape)
    assert (onnx_frames == ref_frames).all(), int((onnx_frames != ref_frames).sum())
    print('ok onnx_generate == Decoder.generate', onnx_frames.shape)
    for f in (enc_file, dec_file, init_file, step_file):
        f.unlink()

    # --- server: socket events, log, files
    import rbf_server_app
    app, sio = rbf_server_app.create_app()
    client = sio.test_client(app)
    client.emit('bootstrap')
    received = client.get_received()
    boot = [m for m in received if m['name'] == 'bootstrap'][0]['args'][0]
    assert boot['model_selection']['timestamp'], 'a loaded checkpoint is needed for the server test'
    controller_file = None
    client.emit('onnx_get')
    snap = [m for m in client.get_received() if m['name'] == 'onnx_update'][-1]['args'][0]
    assert snap['state'] == 'idle' and snap['log'] == [], snap

    client.emit('onnx_export', {})
    updates = []
    deadline = time.time() + 240
    final = None
    while time.time() < deadline:
        for m in client.get_received():
            if m['name'] == 'onnx_export_result':
                assert m['args'][0]['ok'], m['args'][0]
                controller_file = m['args'][0]['checkpoint']
            if m['name'] == 'onnx_update':
                updates.append(m['args'][0])
                if m['args'][0]['state'] in ('done', 'failed'):
                    final = m['args'][0]
        if final:
            break
        time.sleep(0.2)
    assert final is not None, 'export did not finish'
    assert final['state'] == 'done', final
    assert any(u['state'] == 'running' for u in updates) and len(updates) > 3, len(updates)
    assert controller_file and controller_file.endswith('.pt')
    base = controller_file[:-3]
    out = tmp / 'models' / BRANCH / 'onnx'
    for suffix in ('encoder', 'decoder', 'decoder_init', 'decoder_step'):
        assert (out / f'{base}_{suffix}.onnx').is_file(), (suffix, os.listdir(out))
    assert final['log'][-1].startswith('Done.') and str(out) in final['log'][-1], final['log'][-3:]
    print('ok server export:', controller_file, f'({len(updates)} updates)')

    # --- a second start while nothing runs works; a missing checkpoint is reported
    ok, err = rbf_server_app.OnnxExportManager().start('model_2000-01-01_00-00-00.pt')
    assert not ok and 'not found' in err, (ok, err)
    print('ok missing checkpoint')
    print('ALL OK')
finally:
    shutil.rmtree(tmp, ignore_errors=True)
