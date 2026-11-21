"""Tests for the training speed / stability options: fused attention layers, one-pass loss, pre-norm, LR warm-up,
early stopping, 'last' checkpoint interval and the max_seq_len adaption of datasets.

Run from the project root:  python tests/test_training_options.py   (prints "ALL OK")
"""
import glob
import os
import random
import shutil
import sys
from types import SimpleNamespace

os.environ['GITHUB_REF_NAME'] = '_test_options'  # isolates models/, data/ and docs/ from real branches

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from hparams import hparams as default_hp  # noqa: E402
from modules import compound  # noqa: E402
from modules.model import (Decoder, Encoder, FusedDecoderLayer, FusedEncoderLayer, TransformerVAE,  # noqa: E402
                           field_slices_of, vae_loss)
from modules.train_params import ALLOWED_KEYS, coerce_param  # noqa: E402
from modules.training import run_training  # noqa: E402
from modules.utilities import load_dataset, load_model, load_scheduler  # noqa: E402

BRANCH = '_test_options'
DIRS = [os.path.join(ROOT, sub, BRANCH) for sub in ('models', 'data', 'docs')]
DATASET = 'tiny.npz'
TINY = {'d_model': 32, 'n_heads': 4, 'n_layers': 2, 'ff_dim': 64, 'latent_dim': 16, 'n_memory_tokens': 4}


def tiny_hparams(**extra):
    hp = SimpleNamespace(**vars(default_hp))
    for key, value in dict(TINY, **extra).items():
        setattr(hp, key, value)
    return hp


def cleanup():
    for d in DIRS:
        shutil.rmtree(d, ignore_errors=True)


def test_layers_match_torch():
    d, h, ff = 32, 4, 64
    torch.manual_seed(0)
    for norm_first in (False, True):
        ref = nn.TransformerEncoderLayer(d, h, ff, dropout=0.0, activation='gelu', batch_first=True, norm_first=norm_first)
        mine = FusedEncoderLayer(d, h, ff, dropout=0.0, norm_first=norm_first)
        mine.load_state_dict(ref.state_dict())  # identical parameter names
        ref.eval()
        mine.eval()
        x = torch.randn(3, 10, d)
        pad = torch.zeros(3, 10, dtype=torch.bool)
        pad[0, 7:] = True
        pad[1, 5:] = True
        out_ref = ref(x, src_key_padding_mask=pad)
        out_mine = mine(x, (~pad)[:, None, None, :])
        assert torch.allclose(out_ref[~pad], out_mine[~pad], atol=1e-5), f'encoder layer differs (norm_first={norm_first})'

        ref = nn.TransformerDecoderLayer(d, h, ff, dropout=0.0, batch_first=True, norm_first=norm_first)
        mine = FusedDecoderLayer(d, h, ff, dropout=0.0, norm_first=norm_first)
        mine.load_state_dict(ref.state_dict())
        ref.eval()
        mine.eval()
        tgt, mem = torch.randn(2, 9, d), torch.randn(2, 5, d)
        out_ref = ref(tgt, mem, tgt_mask=nn.Transformer.generate_square_subsequent_mask(9))
        out_mine = mine(tgt, mem)
        assert torch.allclose(out_ref, out_mine, atol=1e-5), f'decoder layer differs (norm_first={norm_first})'
    print('ok fused layers == torch layers (post-norm and pre-norm)')


def test_state_dict_keys_unchanged():
    """Old (post-norm, nn.Transformer*) checkpoints must keep loading: same parameter names."""
    hp = tiny_hparams(norm_first=False)
    enc, dec = Encoder(hp), Decoder(hp)
    ref_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(32, 4, 64, batch_first=True, activation='gelu'),
                                    num_layers=2, enable_nested_tensor=False)
    ref_dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(32, 4, 64, batch_first=True), num_layers=2)
    enc_keys = {k[len('encoder.'):] for k in enc.state_dict() if k.startswith('encoder.')}
    dec_keys = {k[len('transformer_decoder.'):] for k in dec.state_dict() if k.startswith('transformer_decoder.')}
    assert enc_keys == set(ref_enc.state_dict()), enc_keys ^ set(ref_enc.state_dict())
    assert dec_keys == set(ref_dec.state_dict()), dec_keys ^ set(ref_dec.state_dict())
    # a checkpoint without norm_first is a post-norm model; pre-norm adds the final norms
    assert not Encoder(SimpleNamespace(**{k: v for k, v in vars(hp).items() if k != 'norm_first'})).norm_first
    pre = Encoder(tiny_hparams(norm_first=True))
    assert any(k.startswith('encoder.norm.') for k in pre.state_dict())
    print('ok state dict keys of post-norm models unchanged')


def reference_loss(logits, targets, slices, weights, pad=0, eos=2):
    """The former per-field loop (4 separate cross entropies)."""
    step = targets[..., 0]
    n = (step != pad).sum().clamp(min=1).float()
    is_eos = step == eos
    recon, ces = 0.0, []
    for f, (s, e) in enumerate(slices):
        t = targets[..., f]
        if f > 0:
            t = t.masked_fill(is_eos, pad)
        ce = F.cross_entropy(logits[..., s:e].reshape(-1, e - s).float(), t.reshape(-1), ignore_index=pad, reduction='sum')
        cnt = (t != pad).sum().clamp(min=1).float()
        recon = recon + weights[f] * ce / n
        ces.append(ce / cnt)
    return recon, torch.stack(ces)


def test_loss_matches_reference():
    hp = tiny_hparams()
    rng = random.Random(3)
    seqs = []
    for _ in range(6):
        notes = [(rng.randrange(32), rng.randrange(54), rng.randrange(32), rng.randrange(33))
                 for _ in range(rng.randrange(0, 30))]
        seqs.append(compound.notes_to_frames(notes, hp))
    targets = torch.from_numpy(np.stack(seqs).astype(np.int64))
    slices = field_slices_of(hp.field_vocab_sizes)
    weights = [1.0, 0.5, 2.0, 0.25]
    torch.manual_seed(0)
    logits_a = torch.randn(targets.shape[0], targets.shape[1], sum(hp.field_vocab_sizes), requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    mu, logvar = torch.zeros(6, 16), torch.zeros(6, 16)
    new = vae_loss(logits_a, targets, mu, logvar, beta=0.0, field_slices=slices, field_weights=weights)
    ref_recon, ref_ce = reference_loss(logits_b, targets, slices, weights)
    assert torch.allclose(new[1], ref_recon, atol=1e-5), (new[1], ref_recon)
    assert torch.allclose(new[5], ref_ce, atol=1e-5), (new[5], ref_ce)
    new[1].backward()
    ref_recon.backward()
    assert torch.allclose(logits_a.grad, logits_b.grad, atol=1e-6), 'gradients differ'
    print('ok one-pass loss == per-field loop (value, field CE, gradient)')


def test_resize_frames_array():
    hp = default_hp
    notes = [(i, 3, 5, 16) for i in range(20)]
    data = np.stack([compound.notes_to_frames(notes, hp), compound.notes_to_frames(notes[:3], hp)])  # [2, 128, 4]
    longer, cut = compound.resize_frames_array(data, 200)
    assert longer.shape == (2, 200, 4) and cut == 0 and (longer[:, 128:] == 0).all()
    same, cut = compound.resize_frames_array(data, 128)
    assert same.shape == data.shape and cut == 0
    short, cut = compound.resize_frames_array(data, 10)
    assert short.shape == (2, 10, 4) and cut == 1, cut  # only the 20-note window loses frames
    assert short[0, -1, 0] == compound.EOS, 'cut sequence ends with EOS'
    assert (short[0, :9] == data[0, :9]).all(), 'first frames stay'
    assert (short[1] == data[1, :10]).all(), 'short sequence unchanged (EOS inside)'
    print('ok resize_frames_array (pad / cut with EOS)')


def test_warmup_scheduler():
    def lrs(ratio, epochs=10, batches=10):
        param = nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr=1e-3)
        hp = SimpleNamespace(learning_rate=1e-3, scheduler_start_lr=1e-3, scheduler_end_lr=1e-4,
                             scheduler_exp_curve_factor=-8.0, n_epochs=epochs, scheduler_warmup_ratio=ratio)
        sched = load_scheduler(opt, [0] * batches, hp)
        out = []
        for _ in range(epochs * batches):
            out.append(opt.param_groups[0]['lr'])
            sched.step()
        return out

    plain = lrs(0.0)
    assert abs(plain[0] - 1e-3) < 1e-12 and abs(plain[-1] - 1e-4) < 1e-9, (plain[0], plain[-1])
    warm = lrs(0.1)  # 10 warm-up steps of 100
    assert abs(warm[0] - 1e-4) < 1e-9, warm[0]  # start at 1/10 of the start LR
    assert all(warm[i] < warm[i + 1] for i in range(9)), 'rising during warm-up'
    assert abs(warm[9] - 1e-3) < 1e-9 and abs(warm[10] - 1e-3) < 1e-9, (warm[9], warm[10])
    assert all(warm[i] >= warm[i + 1] - 1e-12 for i in range(10, 99)), 'decay after the warm-up'
    assert abs(warm[-1] - 1e-4) < 1e-9, warm[-1]
    print('ok LR warm-up (linear up, then the unchanged decay to the end LR)')


def make_dataset(n=64, length=128, seed=0):
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n):
        notes = [(int(rng.integers(32)), int(rng.integers(54)), int(rng.integers(32)), int(rng.integers(33)))
                 for _ in range(int(rng.integers(5, 40)))]
        seqs.append(compound.notes_to_frames(notes, default_hp, max_len=length))
    metadata = np.empty(n, dtype=object)
    for i in range(n):
        metadata[i] = (f'file{i}', 0, 120)
    os.makedirs(os.path.join(ROOT, 'data', BRANCH), exist_ok=True)
    np.savez(os.path.join(ROOT, 'data', BRANCH, DATASET), data=np.stack(seqs).astype(np.int16), metadata=metadata)


def config(**overrides):
    ov = dict(TINY, dataset_name=DATASET, batch_size=16, n_epochs=2, validation_split=0.25, n_samples=0)
    ov.update(overrides)
    return {'mode': 'new', 'overrides': ov, 'options': {'torch_compile': False, 'log_interval': 1}}


def last_epoch():
    files = glob.glob(os.path.join(ROOT, 'models', BRANCH, 'model_last_*.pt'))
    assert files, 'no last checkpoint'
    return torch.load(files[0], map_location='cpu')['epoch']


def test_params_whitelist():
    for key in ('scheduler_warmup_ratio', 'last_checkpoint_interval', 'early_stopping', 'early_stopping_patience',
                'early_stopping_min_delta', 'norm_first', 'max_seq_len'):
        assert key in ALLOWED_KEYS, key
    assert coerce_param('early_stopping', True) is True and coerce_param('norm_first', 'false') is False
    for bad in ('maybe', 2, None):
        try:
            coerce_param('early_stopping', bad)
            raise AssertionError(f'{bad!r} accepted')
        except ValueError:
            pass
    assert coerce_param('last_checkpoint_interval', 3) == 3
    print('ok parameter whitelist (bool coercion)')


def test_dataset_adapts_to_max_seq_len():
    make_dataset(n=12, length=128)
    hp = tiny_hparams(max_seq_len=48, dataset_name=DATASET, n_samples=0, seed=1)
    data, _ = load_dataset(hp, shuffle=False)
    assert data.shape == (12, 48, 4), data.shape
    assert (data[:, :, 0] == compound.EOS).any(axis=1).all(), 'every sequence still has an EOS'
    hp = tiny_hparams(max_seq_len=200, dataset_name=DATASET, n_samples=0, seed=1)
    data, _ = load_dataset(hp, shuffle=False)
    assert data.shape == (12, 200, 4)
    print('ok load_dataset pads / cuts to the max_seq_len of the model')


def test_last_interval_and_early_stopping():
    make_dataset()
    cleanup_models()
    # last checkpoint every 3 epochs: epoch 3 is written, the end (epoch 4) as well
    seen = []
    events = []
    result = run_training(config(n_epochs=4, last_checkpoint_interval=3), on_event=events.append)
    last_events = [e for e in events if e['type'] == 'checkpoint' and e['kind'] == 'last']
    assert len(last_events) == 2, f'expected 2 last writes (epoch 3 and 4), got {len(last_events)}'
    assert last_epoch() == 4 and result['early_stopped'] is False
    cleanup_models()
    # every epoch with interval 1
    events = []
    run_training(config(n_epochs=3, last_checkpoint_interval=1), on_event=events.append)
    assert len([e for e in events if e['type'] == 'checkpoint' and e['kind'] == 'last']) == 3
    cleanup_models()
    # early stopping: constant beta, no improvement can reach min_delta=100 -> stops after patience epochs
    events = []
    result = run_training(
        config(n_epochs=10, last_checkpoint_interval=5, early_stopping=True, early_stopping_patience=2,
               early_stopping_min_delta=100.0, annealing_list=[[0.0, 0.1], [1.0, 0.1]]),
        on_event=events.append)
    epochs = [e for e in events if e['type'] == 'epoch']
    # epoch 1 arms it (reference), epochs 2 and 3 count as "no improvement" -> stop after epoch 3
    assert result['early_stopped'] is True and result['stopped'] is False and len(epochs) == 3, (result, len(epochs))
    assert last_epoch() == 3, 'last is written when early stopping ends the run'
    assert any('Early stopping after epoch 3' in e.get('text', '') for e in events if e['type'] == 'log')
    model, hp = load_model(onnx=False, branch=BRANCH, checkpoint_kind='last')
    assert hp.early_stopping is True and hp.norm_first is True and hp.max_seq_len == 128
    cleanup_models()
    # not armed while beta still grows: ramp over the whole run -> nothing can stop before the end
    events = []
    result = run_training(
        config(n_epochs=4, early_stopping=True, early_stopping_patience=1, early_stopping_min_delta=100.0,
               annealing_list=[[0.0, 0.0], [1.0, 0.5]]), on_event=events.append)
    assert result['early_stopped'] is False and len([e for e in events if e['type'] == 'epoch']) == 4
    print('ok last checkpoint interval and early stopping (armed only after the beta annealing)')


def cleanup_models():
    shutil.rmtree(os.path.join(ROOT, 'models', BRANCH), ignore_errors=True)


def test_resume_keeps_post_norm_and_arch():
    make_dataset()
    cleanup_models()
    # old style model: post-norm, no norm_first / warm-up keys in the checkpoint
    run_training(config(n_epochs=1, norm_first=False, last_checkpoint_interval=1))
    path = glob.glob(os.path.join(ROOT, 'models', BRANCH, 'model_last_*.pt'))[0]
    ckpt = torch.load(path, map_location='cpu')
    for key in ('norm_first', 'scheduler_warmup_ratio', 'early_stopping', 'last_checkpoint_interval'):
        ckpt['hparams'].pop(key, None)
    torch.save(ckpt, path)
    model, hp = load_model(onnx=False, branch=BRANCH, checkpoint_kind='last')
    assert not model.encoder.norm_first and not model.decoder.norm_first, 'checkpoint without norm_first = post-norm'
    events = []
    resume = {'mode': 'resume', 'resume': {'mode': 'last', 'previous': 0}, 'overrides': {
        'n_epochs': 1, 'dataset_name': DATASET, 'validation_split': 0.25, 'batch_size': 16, 'norm_first': True,
        'max_seq_len': 64}, 'options': {'torch_compile': False, 'log_interval': 1}}
    run_training(resume, on_event=events.append)
    logs = ' '.join(e.get('text', '') for e in events if e['type'] == 'status')
    assert 'Architecture parameters come from the checkpoint' in logs and 'norm_first' in logs and 'max_seq_len' in logs
    new_path = sorted(glob.glob(os.path.join(ROOT, 'models', BRANCH, 'model_last_*.pt')), key=os.path.getmtime)[-1]
    new_hp = torch.load(new_path, map_location='cpu')['hparams']
    assert new_hp['norm_first'] is False and new_hp['max_seq_len'] == 128, (new_hp['norm_first'], new_hp['max_seq_len'])
    assert new_hp['scheduler_warmup_ratio'] == 0.0, 'no warm-up when resuming'
    load_model(onnx=False, branch=BRANCH, checkpoint_kind='last', previous=0)  # still loads
    print('ok resume keeps the architecture (post-norm, max_seq_len) of the checkpoint, no warm-up')


def test_vae_pre_norm_trains():
    hp = tiny_hparams(norm_first=True)
    model = TransformerVAE(hp)
    frames = torch.from_numpy(np.stack([compound.notes_to_frames([(1, 2, 3, 16), (4, 5, 6, 16)], hp)]).astype(np.int64))
    logits, mu, logvar = model(frames)
    assert logits.shape == (1, 128, sum(hp.field_vocab_sizes)) and torch.isfinite(logits).all()
    model.eval()
    z = torch.zeros(2, hp.latent_dim)
    out = model.decoder.generate(z, max_len=10)
    assert out.shape[0] == 2 and out.shape[2] == 4
    print('ok pre-norm model forward / generate')


if __name__ == '__main__':
    cleanup()
    try:
        test_layers_match_torch()
        test_state_dict_keys_unchanged()
        test_loss_matches_reference()
        test_resize_frames_array()
        test_warmup_scheduler()
        test_params_whitelist()
        test_vae_pre_norm_trains()
        test_dataset_adapts_to_max_seq_len()
        test_last_interval_and_early_stopping()
        test_resume_keeps_post_norm_and_arch()
        print('ALL OK')
    finally:
        cleanup()
