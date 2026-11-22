"""Integration test: training inside rbf_server_app (TrainingManager, socket events, inference during training).

Uses the Flask-SocketIO test client (no port is opened) and a throw-away branch (`_test_server`) with a tiny model and a
random dataset. hparams.py is never touched: hparams_save writes to a temporary copy.

Run from the project root:  python tests/test_train_server.py
"""
import os
import shutil
import sys
import tempfile
import time

os.environ['GITHUB_REF_NAME'] = '_test_server'  # isolates models/, data/ and docs/ from real branches

import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import rbf_server_app as server  # noqa: E402
from hparams import hparams as default_hp  # noqa: E402
from modules import compound  # noqa: E402
from modules.training import run_training  # noqa: E402

BRANCH = '_test_server'
DIRS = [os.path.join(ROOT, sub, BRANCH) for sub in ('models', 'data', 'docs')]
TINY_MODEL = {'d_model': 32, 'n_heads': 4, 'n_layers': 1, 'ff_dim': 64, 'latent_dim': 16, 'n_memory_tokens': 4}


def cleanup():
    for d in DIRS:
        shutil.rmtree(d, ignore_errors=True)


def make_frames(n=64, seed=0):
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n):
        notes = [(int(rng.integers(32)), int(rng.integers(54)), int(rng.integers(32)), int(rng.integers(33)))
                 for _ in range(int(rng.integers(5, 40)))]
        seqs.append(compound.notes_to_frames(notes, default_hp))
    return np.stack(seqs).astype(np.int16)


def make_dataset(name='tiny.npz', n=64, seed=0):
    """Synthetic compound dataset [n, max_seq_len, 4]."""
    data = make_frames(n, seed)
    metadata = np.empty(n, dtype=object)
    for i in range(n):
        metadata[i] = (f'file{i}', 0, 120)
    os.makedirs(os.path.join(ROOT, 'data', BRANCH), exist_ok=True)
    np.savez(os.path.join(ROOT, 'data', BRANCH, name), data=data, metadata=metadata)


def new_config(**overrides):
    ov = dict(TINY_MODEL, dataset_name='tiny.npz', batch_size=16, n_epochs=2, validation_split=0.25, n_samples=0)
    ov.update(overrides)
    return {'mode': 'new', 'overrides': ov, 'options': {'torch_compile': False, 'log_interval': 1}}


class Harness:
    def __init__(self, hparams_copy):
        self.app, self.sio = server.create_app(sample_count=4, hparams_path=hparams_copy)
        self.http = self.app.test_client()
        self.client = self.sio.test_client(self.app)
        self.received = []

    def drain(self):
        self.received.extend(self.client.get_received())

    def emit_and_get(self, name, payload=None, expect=None):
        self.drain()
        self.received.clear()
        if payload is None:
            self.client.emit(name)
        else:
            self.client.emit(name, payload)
        deadline = time.time() + 120
        while time.time() < deadline:
            self.drain()
            for item in self.received:
                if item['name'] == expect:
                    return item['args'][0]
            time.sleep(0.05)
        raise AssertionError(f'no {expect} event after {name}; got {[r["name"] for r in self.received]}')

    def status(self):
        return self.http.get('/api/train/status').json

    def health(self):
        return self.http.get('/api/health').json

    def wait_state(self, states, timeout=300):
        deadline = time.time() + timeout
        while time.time() < deadline:
            snap = self.status()
            if snap['state'] in states:
                return snap
            time.sleep(0.2)
        raise AssertionError(f'state did not reach {states}: {self.status()["state"]}')

    def wait_for(self, predicate, timeout=120, what='condition'):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return
            time.sleep(0.1)
        raise AssertionError(f'timeout waiting for {what}')


def none_items(h):
    cfg = h.emit_and_get('train_get_config', expect='train_config')
    return len(cfg['model_options']['modes']['none']['items']), cfg


def main():
    cleanup()
    hparams_copy_dir = tempfile.mkdtemp(prefix='hparams_copy_')
    hparams_copy = os.path.join(hparams_copy_dir, 'hparams.py')
    shutil.copy(os.path.join(ROOT, 'src', 'hparams.py'), hparams_copy)
    try:
        make_dataset()
        legacy = np.zeros((4, 256), dtype=np.int64)
        np.savez(os.path.join(ROOT, 'data', BRANCH, 'legacy.npz'), data=legacy,
                 metadata=np.array([('l.mid', 0, 120)] * 4, dtype=object))
        # A model must exist so that the server starts with working inference.
        run_training(new_config(n_epochs=1))
        time.sleep(1.1)

        h = Harness(hparams_copy)
        boot = h.emit_and_get('bootstrap', expect='bootstrap')
        assert boot['app_status']['ready'] is True and boot['app_status']['training'] is False, boot['app_status']
        items_before, cfg = none_items(h)
        assert cfg['datasets'] == ['tiny.npz'] and cfg['defaults']['batch_size'] == default_hp.batch_size
        assert cfg['legacy_datasets'] == ['legacy.npz'], cfg['legacy_datasets']
        assert 'legacy.npz' not in boot['dataset_options'], boot['dataset_options']
        assert cfg['state']['state'] == 'idle'
        print('ok server starts with model, train_get_config works')

        # ---- validation errors, no process is started
        result = h.emit_and_get('train_start', new_config(d_model=30, n_heads=4), expect='train_start_result')
        assert result['ok'] is False and 'd_model' in result['errors'] and 'n_heads' in result['errors'], result
        result = h.emit_and_get('train_start', new_config(dataset_name='missing.npz', learning_rate=-1, n_epochs=0,
                                                          annealing_list=[[0.5, 0.1], [0.2, 0.3]]),
                                expect='train_start_result')
        assert not result['ok'] and {'dataset_name', 'learning_rate', 'n_epochs', 'annealing_list'} <= set(result['errors']), result
        result = h.emit_and_get('train_start', {'mode': 'resume', 'resume': {'mode': 'best_val', 'previous': 9}},
                                expect='train_start_result')
        assert not result['ok'] and 'resume_mode' in result['errors'], result
        result = h.emit_and_get('train_start', new_config(dataset_name='legacy.npz'), expect='train_start_result')
        assert not result['ok'] and 'legacy triplet' in result['errors']['dataset_name'], result
        assert h.status()['state'] == 'idle'
        print('ok invalid configs rejected per field, no worker started')

        # ---- full run: inference stays available during training, updates broadcast, new checkpoint appears afterwards
        result = h.emit_and_get('train_start', new_config(n_epochs=2), expect='train_start_result')
        assert result['ok'] is True, result
        health = h.health()
        assert health['training'] is True and health['ready'] is True and health['model_available'] is True, health
        boot = h.emit_and_get('bootstrap', expect='bootstrap')  # inference keeps working while the worker trains
        assert boot['app_status']['training'] is True and boot['app_status']['ready'] is True, boot['app_status']
        assert h.health()['model_available'] is True
        second = h.emit_and_get('train_start', new_config(), expect='train_start_result')
        assert second['ok'] is False and 'already running' in second['errors']['general'], second
        snap = h.wait_state(('finished', 'failed'))
        assert snap['state'] == 'finished' and snap['stopped'] is False, snap['error']
        assert len(snap['epochs']) == 2 and snap['epochs'][-1]['val'] is not None
        assert set(snap['epochs'][-1]['train']['fields']) == {'step', 'pitch', 'velocity', 'deviation'}, snap['epochs'][-1]
        assert set(snap['epochs'][-1]['val']['fields']) == {'step', 'pitch', 'velocity', 'deviation'}
        assert any(c['kind'] == 'final' for c in snap['checkpoints'])
        h.drain()
        updates = [r['args'][0] for r in h.received if r['name'] == 'train_update']
        h.wait_for(lambda: (h.drain() or True) and any(r['name'] == 'train_update' and r['args'][0]['state'] == 'finished'
                                                       for r in h.received), what='final train_update')
        health = h.health()
        assert health['ready'] is True and health['training'] is False and health['model_available'] is True, health
        items_after, _ = none_items(h)
        assert items_after == items_before + 1, (items_before, items_after)
        print('ok full run: inference available during training, new checkpoint in the catalog afterwards')

        # ---- stop
        time.sleep(1.1)
        result = h.emit_and_get('train_start', new_config(n_epochs=200), expect='train_start_result')
        assert result['ok'] is True
        h.wait_for(lambda: h.status()['last_batch'] is not None, timeout=120, what='first batch')
        stop = h.emit_and_get('train_stop', expect='train_stop_result')
        assert stop['ok'] is True
        snap = h.wait_state(('finished', 'failed'), timeout=120)
        assert snap['state'] == 'finished' and snap['stopped'] is True, snap
        assert any(c['kind'] == 'last' for c in snap['checkpoints'])
        h.wait_for(lambda: h.health()['ready'], what='inference back after stop')
        assert h.emit_and_get('train_stop', expect='train_stop_result')['ok'] is False
        print('ok stop: stopped=True, last checkpoint written, inference back')

        # ---- resume from the last checkpoint
        time.sleep(1.1)
        resume_cfg = {
            'mode': 'resume',
            'resume': {'mode': 'last', 'previous': 0, 'continue_beta': True, 'end_lr_factor': 0.1},
            'overrides': {'n_epochs': 1, 'validation_split': 0.0, 'd_model': 64},
            'options': {'torch_compile': False, 'log_interval': 1},
        }
        result = h.emit_and_get('train_start', resume_cfg, expect='train_start_result')
        assert result['ok'] is True, result
        snap = h.wait_state(('finished', 'failed'))
        assert snap['state'] == 'finished', snap['error']
        assert len(snap['epochs']) == 1 and snap['epochs'][0]['val'] is None
        h.wait_for(lambda: h.health()['ready'], what='inference back after resume')
        print('ok resume via server')

        # ---- worker crash: server stays up, inference comes back
        time.sleep(1.1)
        # a compound dataset without the metadata array passes the format check but crashes the worker
        np.savez(os.path.join(ROOT, 'data', BRANCH, 'broken.npz'), data=make_frames(8, 3))
        result = h.emit_and_get('train_start', new_config(dataset_name='broken.npz'), expect='train_start_result')
        assert result['ok'] is True, result
        snap = h.wait_state(('finished', 'failed'))
        assert snap['state'] == 'failed' and snap['error'], snap
        h.wait_for(lambda: h.health()['ready'], what='inference back after crash')
        assert h.http.get('/').status_code == 200
        print('ok worker crash: state=failed with message, server and inference still up:', snap['error'][:80])

        # ---- hparams_save writes to the temporary copy only
        original = open(os.path.join(ROOT, 'src', 'hparams.py'), 'rb').read()
        saved = h.emit_and_get('hparams_save', {'changes': {'learning_rate': 0.002, 'n_epochs': 7}}, expect='hparams_save_result')
        assert saved['ok'] is True and set(saved['changed']) == {'learning_rate', 'n_epochs'}, saved
        text = open(hparams_copy, encoding='utf-8').read()
        assert "'learning_rate': 0.002" in text and "'n_epochs': 7" in text
        rejected = h.emit_and_get('hparams_save', {'changes': {'pad_token': 5}}, expect='hparams_save_result')
        assert rejected['ok'] is False and 'pad_token' in rejected['errors']
        assert open(os.path.join(ROOT, 'src', 'hparams.py'), 'rb').read() == original, 'real hparams.py untouched'
        print('ok hparams_save (temporary copy), invalid key rejected')

        print('ALL OK')
    finally:
        cleanup()
        shutil.rmtree(hparams_copy_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
