"""End-to-end test of the compound pipeline inside rbf_server_app (no browser, no port).

Idle start without model/dataset -> MIDI extraction from midifiles/ (Dataset tab) -> mini training (Training tab) ->
bootstrap ready -> realtime decode -> full-field render export -> Encode MIDI. Uses the Flask-SocketIO test client
and a throw-away branch (`_test_pipeline`) that is removed at the end.

Run from the project root:  python tests/test_compound_pipeline.py
"""
import os
import shutil
import sys
import time
from pathlib import Path

BRANCH = '_test_pipeline'
os.environ['GITHUB_REF_NAME'] = BRANCH

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np  # noqa: E402

import rbf_server_app as server  # noqa: E402
from modules import compound  # noqa: E402

DIRS = [ROOT / sub / BRANCH for sub in ('models', 'data', 'docs', 'outputs')]
TINY_MODEL = {'d_model': 64, 'n_heads': 4, 'n_layers': 2, 'ff_dim': 128, 'latent_dim': 32, 'n_memory_tokens': 4}


class Harness:
    def __init__(self):
        self.app, self.sio = server.create_app(sample_count=4)
        self.http = self.app.test_client()
        self.client = self.sio.test_client(self.app)
        self.received = []

    def drain(self):
        self.received.extend(self.client.get_received())

    def ask(self, event, payload, result_event, timeout=120):
        self.drain()
        self.received.clear()
        if payload is None:
            self.client.emit(event)
        else:
            self.client.emit(event, payload)
        return self.wait(result_event, timeout=timeout)

    def wait(self, name, predicate=None, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.drain()
            for item in self.received:
                if item['name'] == name and (predicate is None or predicate(item['args'][0])):
                    return item['args'][0]
            time.sleep(0.05)
        raise AssertionError(f'no {name} event; got {sorted({r["name"] for r in self.received})}')

    def frame(self, boot, **buttons):
        payload = {
            'cursor': [0.05, -0.05],
            'draggable_points': boot['draggable_points'],
            'slider_values': {'gridResolution': 4, 'sigma': 0.4, 'sampleCount': 4},
            'button_values': dict({'datasetName': boot['dataset_selection'], 'decodeEnabled': False,
                                   'renderEnabled': False, 'encodeMidi': False}, **buttons.pop('button_values', {})),
        }
        payload.update(buttons)
        return payload


def main():
    for d in DIRS:
        shutil.rmtree(d, ignore_errors=True)
    try:
        h = Harness()

        # ---- idle without model and dataset
        boot = h.ask('bootstrap', None, 'bootstrap')
        assert boot['app_status']['ready'] is False and not boot['app_status']['model_available'], boot['app_status']
        assert h.http.get('/api/health').json['ok']
        print('ok idle start without model and dataset')

        # ---- extraction from midifiles/ (Dataset tab)
        res = h.ask('ds_start', {'kind': 'extract', 'params': {'folder': str(ROOT / 'midifiles'), 'name': 'groove',
                                                                'mapping': 'groove_to_specific'}}, 'ds_start_result')
        assert res['ok'], res
        final = h.wait('ds_update', lambda s: s['state'] in ('finished', 'failed'), timeout=300)
        assert final['state'] == 'finished', final
        data = np.load(ROOT / 'data' / BRANCH / 'groove.npz', allow_pickle=True)['data']
        assert data.ndim == 3 and data.shape[1:] == (server._DEFAULT_HPARAMS.max_seq_len, 4) and data.dtype == np.int16
        print(f'ok extraction: {data.shape} frames, max notes {int(compound.valid_note_mask(data).sum(1).max())}')

        # ---- mini training through the server (Training tab)
        cfg = {'mode': 'new', 'overrides': dict(TINY_MODEL, dataset_name='groove.npz', batch_size=32, n_epochs=25,
                                                 validation_split=0.1, learning_rate=1e-3, scheduler_start_lr=1e-3,
                                                 scheduler_end_lr=2e-4, annealing_list=[[0.0, 0.0], [1.0, 0.01]]),
               'options': {'torch_compile': False, 'log_interval': 5}}
        res = h.ask('train_start', cfg, 'train_start_result')
        assert res['ok'], res
        deadline = time.time() + 900
        while h.http.get('/api/train/status').json['state'] not in ('finished', 'failed'):
            assert time.time() < deadline, 'training timeout'
            time.sleep(0.5)
        snap = h.http.get('/api/train/status').json
        assert snap['state'] == 'finished', snap.get('error')
        first, last = snap['epochs'][0]['train'], snap['epochs'][-1]['train']
        print(f"ok training: rec {first['rec']:.3f} -> {last['rec']:.3f}, fields {last['fields']}")
        assert last['rec'] < first['rec']

        deadline = time.time() + 60
        while not h.http.get('/api/health').json['ready']:
            assert time.time() < deadline, 'inference did not come back'
            time.sleep(0.2)
        boot = h.ask('bootstrap', None, 'bootstrap')
        assert boot['app_status']['ready'] is True, boot['app_status']
        assert boot['dataset_selection'] == 'groove.npz' and len(boot['draggable_points']) == 4
        print('ok bootstrap ready with compound model and dataset')

        # ---- realtime decode
        h.received.clear()
        h.client.emit('frame', h.frame(boot, button_values={'decodeEnabled': True}))
        done = h.wait('frame', lambda r: r['decode_status'] == 'done', timeout=120)
        matrix = np.asarray(done['decoded_matrix'], dtype=np.float32)
        assert matrix.shape == (54, 32 * 33), matrix.shape
        assert float(matrix.sum()) > 0, 'decoded matrix is empty'
        latent = np.asarray(done['latent_vector'], dtype=np.float32)
        runtime_matrix = h_runtime_matrix(latent)
        assert np.array_equal(runtime_matrix, matrix), 'realtime decoder differs from batched decoding'
        print(f'ok realtime decode: {int((np.diff(np.concatenate([[0], (matrix > 0).any(0).astype(int)])) > 0).sum())} onsets, '
              'identical to the batched decoder')

        # ---- full-field render (4 x 4) with export
        h.received.clear()
        h.client.emit('frame', h.frame(boot, button_values={'renderEnabled': True}))
        done = h.wait('frame', lambda r: r['render_status'] in ('done', 'error'), timeout=300)
        assert done['render_status'] == 'done', done['render_error']
        export = np.load(done['render_output_path'], allow_pickle=True)
        seqs = export['sequences']
        assert seqs.shape == (16, server._DEFAULT_HPARAMS.max_seq_len, 4) and seqs.dtype == np.int16, seqs.shape
        assert str(export['token_format']) == 'compound_v1' and 'sequences_1d' not in export.files
        assert list(export['field_vocab_sizes']) == [35, 57, 35, 36] and int(export['n_specials']) == 3
        for row in seqs:
            check_sorted(row)
        print(f'ok render export {Path(done["render_output_path"]).name}: sequences {seqs.shape}, '
              f'{int(compound.valid_note_mask(seqs).sum())} notes')

        # ---- Encode MIDI: matrix and frames give a latent of length latent_dim; flat token lists are rejected
        frames = data[0][:compound.sequence_length(data[0])].astype(int).tolist()
        for buffer in ({'matrix': compound.frames_to_matrix(data[0]).tolist()}, {'frames': frames}, frames[:-1]):
            h.received.clear()
            h.client.emit('frame', h.frame(boot, midi_buffer=buffer, button_values={'encodeMidi': True}))
            r = h.wait('frame')
            assert len(r['latent_vector']) == TINY_MODEL['latent_dim'], len(r['latent_vector'])
        h.received.clear()
        h.client.emit('frame', h.frame(boot, midi_buffer=[10, 70, 106, 3, 2], button_values={'encodeMidi': True}))
        err = h.wait('error')
        assert 'no longer supported' in err['message'], err
        print('ok Encode MIDI with matrix / frames, flat triplet list rejected')
        print('ALL OK')
    finally:
        for d in DIRS:
            shutil.rmtree(d, ignore_errors=True)


def h_runtime_matrix(latent):
    """Decode the latent once more with the batched runtime path (same checkpoint as the server)."""
    runtime = server.CudaRBFRuntime(previous=0, checkpoint_kind=None)
    return runtime.decode_latent_to_matrix(latent)


def check_sorted(frames):
    raw = compound.frames_to_raw(frames)
    keys = [tuple(x) for x in raw[:, :2].tolist()]
    assert keys == sorted(keys) and len(set(keys)) == len(keys), 'frames not sorted by (step, pitch) / duplicates'


if __name__ == '__main__':
    main()
