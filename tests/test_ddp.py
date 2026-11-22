"""Tests for multi-GPU training (DistributedDataParallel), run with CPU processes (gloo) instead of GPUs.

Covers: GPU selection helpers, TensorBatcher sharding, a full DDP run through train_worker.py (events, checkpoints
loadable), synchronised stop, and a killed rank. Uses a throw-away branch folder that is removed at the end.
Run from the project root:  python tests/test_ddp.py
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

os.environ['GITHUB_REF_NAME'] = '_test_ddp'  # isolates models/, data/ and docs/ from real branches

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from hparams import hparams as default_hp  # noqa: E402
from modules import compound  # noqa: E402
from modules.distributed import device_roles, normalize_gpus, plan_devices  # noqa: E402
from modules.training import TensorBatcher  # noqa: E402
from modules.utilities import load_model  # noqa: E402

BRANCH = '_test_ddp'
DIRS = [os.path.join(ROOT, sub, BRANCH) for sub in ('models', 'data', 'docs')]
DATASET = 'tiny.npz'
WORKER = os.path.join(ROOT, 'src', 'train_worker.py')
TINY_MODEL = {'d_model': 32, 'n_heads': 4, 'n_layers': 1, 'ff_dim': 64, 'latent_dim': 16, 'n_memory_tokens': 4}


def make_dataset(n=64, seed=0):
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n):
        notes = [(int(rng.integers(32)), int(rng.integers(54)), int(rng.integers(32)), int(rng.integers(33)))
                 for _ in range(int(rng.integers(5, 40)))]
        seqs.append(compound.notes_to_frames(notes, default_hp))
    metadata = np.empty(n, dtype=object)
    for i in range(n):
        metadata[i] = (f'file{i}', 0, 120)
    os.makedirs(os.path.join(ROOT, 'data', BRANCH), exist_ok=True)
    np.savez(os.path.join(ROOT, 'data', BRANCH, DATASET), data=np.stack(seqs).astype(np.int16), metadata=metadata)


def config(n_epochs=2, processes=2, **overrides):
    ov = dict(TINY_MODEL, dataset_name=DATASET, batch_size=16, n_epochs=n_epochs, validation_split=0.25, n_samples=0,
              last_checkpoint_interval=1)
    ov.update(overrides)
    return {'mode': 'new', 'overrides': ov,
            'options': {'torch_compile': False, 'log_interval': 1, 'ddp_cpu_processes': processes}}


def cleanup():
    for d in DIRS:
        shutil.rmtree(d, ignore_errors=True)


def test_gpu_helpers():
    assert normalize_gpus(None, 4) == [0]
    assert normalize_gpus([], 4) == [0]
    assert normalize_gpus([3, 1, 1, 9, 'x', -1], 4) == [1, 3]
    assert normalize_gpus([0, 1], 0) == []
    assert len(plan_devices({'ddp_cpu_processes': 3})) == 3
    # automatic split: the last GPU is kept for inference, the others train
    assert device_roles(0) == {'train_gpus': [], 'inference_gpu': None, 'shared': True}
    assert device_roles(1) == {'train_gpus': [0], 'inference_gpu': 0, 'shared': True}
    assert device_roles(2) == {'train_gpus': [0], 'inference_gpu': 1, 'shared': False}
    assert device_roles(4) == {'train_gpus': [0, 1, 2], 'inference_gpu': 3, 'shared': False}
    from rbf_server_app import validate_train_config
    for count, expected in ((0, []), (1, [0]), (2, [0]), (4, [0, 1, 2])):
        clean, errors = validate_train_config({'options': {'gpus': [1, 2, 3]}}, ['x.npz'], lambda m: 0, {}, gpu_count=count)
        assert clean['options']['gpus'] == expected and 'gpus' not in errors, (count, clean['options'], errors)
    assert len(plan_devices({})) == 1, 'default is a single device'
    print('ok gpu helpers')


def test_batcher_sharding():
    data = torch.arange(50).view(50, 1)
    for world in (2, 3):
        batchers = [TensorBatcher(data, 16, shuffle=True, seed=7, rank=r, world_size=world) for r in range(world)]
        assert len({len(b) for b in batchers}) == 1, 'same number of batches on every rank'
        for _epoch in range(2):
            shards = [list(iter(b)) for b in batchers]
            for step in range(len(batchers[0])):
                sizes = {s[step].shape[0] for s in shards}
                assert len(sizes) == 1, f'equal batch size per step {sizes}'
            seen = [int(x) for s in shards for b in s for x in b.view(-1)]
            assert len(seen) == len(set(seen)), 'no sample used twice in an epoch'
            assert 50 - len(seen) < world * (len(batchers[0]) + 1), 'at most world_size-1 samples dropped per global batch'
            assert len(seen) >= 50 - (world - 1) * 4, f'too many samples dropped: {len(seen)}'
        assert len(batchers[0]) == len(list(iter(batchers[0])))
    single = TensorBatcher(data, 16, shuffle=False)
    assert len(single) == 4 and torch.cat(list(single)).view(-1).tolist() == list(range(50))
    print('ok batcher sharding')


def start_worker(cfg, stop_file=None):
    tmp = tempfile.mkdtemp(prefix='ddp_test_')
    cfg_path = os.path.join(tmp, 'config.json')
    json.dump(cfg, open(cfg_path, 'w'))
    stop_path = stop_file or os.path.join(tmp, 'STOP')
    proc = subprocess.Popen([sys.executable, WORKER, '--config', cfg_path, '--stop-file', stop_path],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            cwd=os.path.join(ROOT, 'src'), env=dict(os.environ, PYTHONUNBUFFERED='1'))
    events, stderr_lines = [], []
    threading.Thread(target=lambda: stderr_lines.extend(proc.stderr), daemon=True).start()

    def read():
        for line in proc.stdout:
            try:
                events.append(json.loads(line))
            except ValueError:
                pass

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    return proc, events, stderr_lines, stop_path, reader, tmp


def wait_for(predicate, timeout, what):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f'timeout waiting for {what}')


def child_pids(pid):
    out = subprocess.run(['pgrep', '-P', str(pid)], capture_output=True, text=True).stdout.split()
    return [int(x) for x in out]


def test_full_ddp_run():
    proc, events, err, _stop, reader, tmp = start_worker(config())
    try:
        code = proc.wait(timeout=240)
        reader.join(5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert code == 0, (code, ''.join(err)[-2000:])
    types = [e['type'] for e in events]
    assert types[-1] == 'done' and 'error' not in types, (types, ''.join(err)[-2000:])
    epochs = [e for e in events if e['type'] == 'epoch']
    assert len(epochs) == 2 and all(e['world_size'] == 2 for e in epochs), epochs
    assert epochs[-1]['train']['loss'] > 0 and epochs[-1]['val'] is not None
    assert len([e for e in events if e['type'] == 'checkpoint' and e['kind'] == 'last']) == 2, 'only rank 0 saves'
    steps = [e['step'] for e in events if e['type'] == 'batch']
    assert max(steps) == 3, f'48 train samples / global batch 16 = 3 steps, got {steps}'
    model, hp = load_model(onnx=False, checkpoint_kind='last', branch=BRANCH)
    assert hp.d_model == 32
    assert not any('module.' in k for k in model.state_dict()), 'no DDP key prefix in the checkpoint'
    assert len(glob.glob(os.path.join(ROOT, 'docs', BRANCH, 'training_log_*.csv'))) == 1, 'one CSV log'
    assert not glob.glob(os.path.join(ROOT, 'models', BRANCH, '*.tmp'))
    print('ok full ddp run')


def test_ddp_stop():
    proc, events, err, stop_path, reader, tmp = start_worker(config(n_epochs=400))
    try:
        wait_for(lambda: any(e['type'] == 'epoch' for e in events), 120, 'first epoch')
        started = time.time()
        open(stop_path, 'w').close()
        code = proc.wait(timeout=120)
        reader.join(5)
        elapsed = time.time() - started
    finally:
        if proc.poll() is None:
            proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)
    assert code == 0, (code, ''.join(err)[-2000:])
    done = events[-1]
    assert done['type'] == 'done' and done['stopped'] is True, done
    assert os.path.exists(done['result']['last_path'])
    print(f'ok ddp stop in {elapsed:.1f}s after {done["result"]["epochs_completed"]} epoch(s)')


def test_ddp_killed_rank():
    proc, events, err, _stop, reader, tmp = start_worker(config(n_epochs=2000))
    try:
        wait_for(lambda: any(e['type'] == 'epoch' for e in events), 120, 'first epoch')
        kids = child_pids(proc.pid)
        assert len(kids) == 1, f'one extra rank expected, found {kids}'
        os.kill(kids[0], 9)
        code = proc.wait(timeout=60)
        reader.join(5)
    finally:
        if proc.poll() is None:
            proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)
    assert code != 0, 'worker must fail when a rank dies'
    assert events[-1]['type'] == 'error' and 'rank' in events[-1]['message'].lower(), events[-1]
    time.sleep(0.5)
    assert not child_pids(proc.pid), 'no process left behind'
    print('ok killed rank is reported')


def main():
    cleanup()
    try:
        make_dataset()
        test_gpu_helpers()
        test_batcher_sharding()
        test_full_ddp_run()
        test_ddp_stop()
        test_ddp_killed_rank()
    finally:
        cleanup()
    print('ALL OK')


if __name__ == '__main__':
    main()
