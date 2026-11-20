"""Tests for the shared training core (modules/training.run_training): events, checkpoints, stop and resume.

Runs on a tiny model and a random dataset in a throw-away branch folder (models/_test_training etc.),
which is removed at the end. Run from the project root:  python tests/test_training_stop.py
"""
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

os.environ['GITHUB_REF_NAME'] = '_test_training'  # isolates models/, data/ and docs/ from real branches

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from hparams import hparams as default_hp  # noqa: E402
from modules.training import TensorBatcher, run_training  # noqa: E402
from modules.utilities import load_model  # noqa: E402

BRANCH = '_test_training'
DIRS = [os.path.join(ROOT, sub, BRANCH) for sub in ('models', 'data', 'docs')]
DATASET = 'tiny.npz'

TINY_MODEL = {'d_model': 32, 'n_heads': 4, 'n_layers': 1, 'ff_dim': 64, 'latent_dim': 16, 'n_memory_tokens': 4}


def make_dataset(n=64, seed=0):
    rng = np.random.default_rng(seed)
    data = np.zeros((n, default_hp.max_seq_len), dtype=np.int64)
    for i in range(n):
        length = int(rng.integers(20, 60))
        data[i, :length] = rng.integers(default_hp.step_token, default_hp.vocab_size, size=length)
        data[i, length] = default_hp.eos_token
    metadata = np.empty(n, dtype=object)
    for i in range(n):
        metadata[i] = (f'file{i}', 0, 120)
    os.makedirs(os.path.join(ROOT, 'data', BRANCH), exist_ok=True)
    np.savez(os.path.join(ROOT, 'data', BRANCH, DATASET), data=data, metadata=metadata)


def base_config(**overrides):
    ov = dict(TINY_MODEL, dataset_name=DATASET, batch_size=16, n_epochs=2, validation_split=0.25, n_samples=0)
    ov.update(overrides)
    return {'mode': 'new', 'overrides': ov, 'options': {'torch_compile': False, 'log_interval': 1}}


def model_files(prefix):
    return sorted(glob.glob(os.path.join(ROOT, 'models', BRANCH, f'{prefix}*.pt')))


def cleanup():
    for d in DIRS:
        shutil.rmtree(d, ignore_errors=True)


def test_tensor_batcher():
    data = torch.arange(50).view(50, 1)
    batcher = TensorBatcher(data, batch_size=16, shuffle=True, seed=1)
    assert len(batcher) == 4
    seen = torch.cat([b for b in batcher]).view(-1).tolist()
    assert sorted(seen) == list(range(50)), 'every sample exactly once per epoch'
    assert seen != list(range(50)), 'shuffled'
    second = torch.cat([b for b in batcher]).view(-1).tolist()
    assert second != seen, 'new order every epoch'
    plain = torch.cat([b for b in TensorBatcher(data, 16, shuffle=False)]).view(-1).tolist()
    assert plain == list(range(50))
    print('ok tensor batcher')


def test_full_run():
    events = []
    result = run_training(base_config(), on_event=events.append)
    types = [e['type'] for e in events]
    assert types.count('epoch') == 2, types
    assert 'batch' in types and 'checkpoint' in types
    epoch_events = [e for e in events if e['type'] == 'epoch']
    assert epoch_events[-1]['val'] is not None
    assert epoch_events[0]['train']['loss'] > 0
    assert result['stopped'] is False and result['epochs_completed'] == 2
    kinds = {e['kind'] for e in events if e['type'] == 'checkpoint'}
    assert {'last', 'best_train', 'final'} <= kinds, kinds
    assert model_files('model_best_val_'), 'best_val checkpoint written'
    assert not glob.glob(os.path.join(ROOT, 'models', BRANCH, '*.tmp')), 'no half-written files'
    model, hp = load_model(onnx=False, checkpoint_kind='last', branch=BRANCH)
    assert hp.d_model == 32 and hp.latent_dim == 16
    model, hp = load_model(onnx=False, branch=BRANCH, name=os.path.basename(result['model_path']))
    steps = [e['step'] for e in events if e['type'] == 'batch']
    assert max(steps) == 3, f'48 train samples / 16 = 3 batches, got {steps}'
    print('ok full run, global_step =', result['global_step'])
    return result


def test_stop():
    state = {'stop': False}
    events = []

    def on_event(event):
        events.append(event)
        if event['type'] == 'batch':
            state['stop'] = True

    started = time.time()
    result = run_training(base_config(n_epochs=5), on_event=on_event, should_stop=lambda: state['stop'])
    assert result['stopped'] is True
    assert result['epochs_completed'] == 0
    assert result['global_step'] == 1, 'stopped right after the first batch'
    assert time.time() - started < 120
    assert not glob.glob(os.path.join(ROOT, 'models', BRANCH, '*.tmp'))
    assert os.path.exists(result['last_path']) and os.path.exists(result['model_path'])
    ckpt = torch.load(result['last_path'], map_location='cpu')
    assert ckpt['global_step'] == 1
    load_model(onnx=False, branch=BRANCH, name=os.path.basename(result['last_path']))
    print('ok stop after first batch')
    return result


def test_resume(previous_result):
    time.sleep(1.1)  # checkpoint file names carry a one-second timestamp
    cfg = {
        'mode': 'resume',
        'resume': {'mode': 'last', 'previous': 0, 'continue_beta': True, 'end_lr_factor': 0.1},
        'overrides': {'n_epochs': 1, 'validation_split': 0.0, 'scheduler_start_lr': 1e-4,
                      'd_model': 64},  # architecture override must be ignored
        'options': {'torch_compile': False, 'log_interval': 1},
    }
    events = []
    before = torch.load(previous_result['last_path'], map_location='cpu')
    result = run_training(cfg, on_event=events.append)
    after = torch.load(result['last_path'], map_location='cpu')
    assert after['hparams']['d_model'] == 32, 'architecture comes from the checkpoint'
    assert result['global_step'] > before['global_step'], 'global_step continues'
    assert [e for e in events if e['type'] == 'epoch'][0]['val'] is None
    assert after['optimizer_state_dict']['state'], 'optimizer state present'
    first_step = [e for e in events if e['type'] == 'batch'][0]['step']
    assert first_step == 1
    print('ok resume: global_step', before['global_step'], '->', result['global_step'])


def test_invalid_config():
    for overrides in ({'d_model': 30, 'n_heads': 4}, {'validation_split': 1.5}, {'not_a_key': 1}):
        try:
            run_training(base_config(**overrides))
        except ValueError:
            continue
        raise AssertionError(f'expected ValueError for {overrides}')
    print('ok invalid config rejected')


def run_worker(config, stop_with):
    """Start train_worker.py as a subprocess; stop_with: 'sigterm' | 'stopfile' | None. Returns (events, returncode)."""
    tmp = tempfile.mkdtemp(prefix='worker_test_')
    try:
        cfg_path = os.path.join(tmp, 'config.json')
        stop_path = os.path.join(tmp, 'STOP')
        with open(cfg_path, 'w') as f:
            json.dump(config, f)
        proc = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, 'src', 'train_worker.py'), '--config', cfg_path, '--stop-file', stop_path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=dict(os.environ),
        )
        events = []
        stopped_sent = False
        for line in proc.stdout:
            event = json.loads(line)  # raises if anything but JSON is written to stdout
            events.append(event)
            if event['type'] == 'batch' and stop_with and not stopped_sent:
                stopped_sent = True
                if stop_with == 'sigterm':
                    proc.send_signal(signal.SIGTERM)
                else:
                    open(stop_path, 'w').close()
        proc.wait(timeout=120)
        return events, proc.returncode
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_worker_subprocess():
    for how in ('sigterm', 'stopfile'):
        time.sleep(1.1)
        events, code = run_worker(base_config(n_epochs=20), stop_with=how)
        assert code == 0, (how, code)
        assert events[-1]['type'] == 'done' and events[-1]['stopped'] is True, events[-1]
        assert any(e['type'] == 'checkpoint' and e['kind'] == 'last' for e in events)
        last = [e for e in events if e['type'] == 'checkpoint' and e['kind'] == 'last'][-1]['path']
        assert os.path.exists(last)
        assert not glob.glob(os.path.join(ROOT, 'models', BRANCH, '*.tmp'))
        print(f'ok worker stops cleanly via {how}')

    time.sleep(1.1)
    events, code = run_worker(base_config(n_epochs=1), stop_with=None)
    assert code == 0 and events[-1]['type'] == 'done' and events[-1]['stopped'] is False
    assert any(e['type'] == 'epoch' for e in events)
    print('ok worker full run')

    events, code = run_worker(base_config(d_model=30, n_heads=4), stop_with=None)
    assert code == 1 and events[-1]['type'] == 'error' and 'divisible' in events[-1]['message'], events[-1]
    print('ok worker reports errors as event')


if __name__ == '__main__':
    cleanup()
    try:
        make_dataset()
        test_tensor_batcher()
        test_invalid_config()
        full = test_full_run()
        time.sleep(1.1)
        stopped = test_stop()
        test_resume(stopped)
        test_worker_subprocess()
        print('ALL OK')
    finally:
        cleanup()
