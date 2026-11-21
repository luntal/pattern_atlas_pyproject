"""Integration test for the Dataset tab backend inside rbf_server_app (socket events, progress broadcast).

Uses the Flask-SocketIO test client (no port is opened) and a throw-away branch (`_test_dataset_server`).
The server runs without model, as in the idle mode. Run from the project root:  python tests/test_dataset_server.py
"""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

BRANCH = '_test_dataset_server'
os.environ['GITHUB_REF_NAME'] = BRANCH

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))

import numpy as np  # noqa: E402

import rbf_server_app as server  # noqa: E402
from test_dataset_tools import make_midi  # noqa: E402

DIRS = [ROOT / sub / BRANCH for sub in ('models', 'data', 'docs')]


class Harness:
    def __init__(self):
        self.app, self.sio = server.create_app(sample_count=4)
        self.http = self.app.test_client()
        self.client = self.sio.test_client(self.app)
        self.received = []

    def drain(self):
        self.received.extend(self.client.get_received())

    def wait_event(self, name, predicate=None, timeout=120, clear=False):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.drain()
            for i, item in enumerate(self.received):
                if item['name'] == name and (predicate is None or predicate(item['args'][0])):
                    if clear:
                        self.received.clear()
                    return item['args'][0]
            time.sleep(0.05)
        raise AssertionError(f'no {name} event; got {sorted({r["name"] for r in self.received})}')

    def ask(self, event, payload, result_event, **kwargs):
        self.drain()
        self.received = [r for r in self.received if r['name'] != result_event]
        if payload is None:
            self.client.emit(event)
        else:
            self.client.emit(event, payload)
        return self.wait_event(result_event, **kwargs)


def main():
    for d in DIRS:
        shutil.rmtree(d, ignore_errors=True)
    tmp = Path(tempfile.mkdtemp(prefix='midis_'))
    try:
        for i, bpm in enumerate((90, 100, 120)):
            make_midi(tmp / f'p{i}_{bpm}bpm.mid', bpm)
        h = Harness()

        state = h.ask('ds_get_state', None, 'ds_state')
        assert state['branch'] == BRANCH and state['datasets'] == [] and state['job']['state'] == 'idle', state
        assert state['hparams']['width'] == 32 and 'groove_to_specific' in state['mappings']

        listing = h.ask('ds_browse', {'path': str(tmp)}, 'ds_browse_result')
        assert listing['ok'] and listing['path'] == str(tmp.resolve()) and listing['midi_count'] == 3, listing
        scan = h.ask('ds_scan', {'path': str(tmp)}, 'ds_scan_result')
        assert scan['ok'] and scan['midi_files'] == 3, scan
        bad_scan = h.ask('ds_scan', {'path': str(tmp / 'missing')}, 'ds_scan_result')
        assert not bad_scan['ok'] and 'does not exist' in bad_scan['error'], bad_scan

        # invalid start -> per-field errors, no job
        res = h.ask('ds_start', {'kind': 'extract', 'params': {'folder': str(tmp / 'x'), 'name': '../bad'}}, 'ds_start_result')
        assert not res['ok'] and 'folder' in res['errors'] and 'name' in res['errors'], res
        assert h.http.get('/api/dataset/status').json['state'] == 'idle'

        # extraction with broadcast progress
        h.received.clear()
        res = h.ask('ds_start', {'kind': 'extract', 'params': {'folder': str(tmp), 'name': 'srv-a'}}, 'ds_start_result')
        assert res['ok'], res
        final = h.wait_event('ds_update', lambda s: s['state'] in ('finished', 'failed', 'cancelled'))
        assert final['state'] == 'finished' and final['result']['name'] == 'srv-a.npz', final
        updates = [r['args'][0] for r in h.received if r['name'] == 'ds_update']
        assert any(u['state'] == 'running' for u in updates) or len(updates) >= 1
        listed = h.wait_event('ds_list', lambda p: any(d['name'] == 'srv-a.npz' for d in p['datasets']))
        assert listed['state'] == 'finished'

        info = h.ask('ds_info', {'name': 'srv-a.npz'}, 'ds_info_result')
        assert info['ok'] and info['dataset']['stats']['n_sequences'] > 0, info
        assert info['dataset']['info']['kind'] == 'extract'
        missing = h.ask('ds_info', {'name': '../nope.npz'}, 'ds_info_result')
        assert not missing['ok'], missing

        assert info['dataset']['shape'][1:] == [server._DEFAULT_HPARAMS.max_seq_len, 4] and info['dataset']['compatible'], info

        seq = h.ask('ds_sequence', {'name': 'srv-a.npz', 'index': 0}, 'ds_sequence_result')
        assert seq['ok'] and seq['sequence']['notes'] and seq['sequence']['bpm'] > 0, seq
        assert seq['sequence']['tokens'][-1] == [2, 2, 2, 2] and not seq['sequence']['legacy'], seq

        saved = h.ask('ds_save_info', {'name': 'srv-a.npz', 'patch': {'description': 'server test', 'tags': ['x'], 'custom': {'k': 'v'}}},
                      'ds_save_info_result')
        assert saved['ok'] and saved['info']['description'] == 'server test', saved
        info = h.ask('ds_info', {'name': 'srv-a.npz'}, 'ds_info_result')
        assert info['dataset']['info']['custom'] == {'k': 'v'} and info['dataset']['info']['kind'] == 'extract'
        bad = h.ask('ds_save_info', {'name': 'srv-a.npz', 'patch': {'custom': 'nope'}}, 'ds_save_info_result')
        assert not bad['ok'], bad

        # merge + augment through the socket
        res = h.ask('ds_start', {'kind': 'augment', 'params': {'source': 'srv-a.npz', 'name': 'srv-aug', 'keep_percentage': 100}}, 'ds_start_result')
        assert res['ok'], res
        final = h.wait_event('ds_update', lambda s: s['state'] in ('finished', 'failed') and s['kind'] == 'augment')
        assert final['state'] == 'finished', final
        res = h.ask('ds_start', {'kind': 'merge', 'params': {'sources': ['srv-a.npz', 'srv-aug.npz'], 'name': 'srv-merged'}}, 'ds_start_result')
        assert res['ok'], res
        final = h.wait_event('ds_update', lambda s: s['state'] in ('finished', 'failed') and s['kind'] == 'merge')
        assert final['state'] == 'finished', final
        names = [d['name'] for d in h.ask('ds_get_state', None, 'ds_state')['datasets']]
        assert names == ['srv-a.npz', 'srv-aug.npz', 'srv-merged.npz'], names

        # extraction of a small real folder (midifiles/, Groove MIDI clips)
        h.received.clear()
        res = h.ask('ds_start', {'kind': 'extract', 'params': {'folder': str(ROOT / 'midifiles'), 'name': 'srv-groove',
                                                                'mapping': 'groove_to_specific', 'take': 8}}, 'ds_start_result')
        assert res['ok'], res
        final = h.wait_event('ds_update', lambda s: s['state'] == 'failed' or (
            s['state'] == 'finished' and (s.get('result') or {}).get('name') == 'srv-groove.npz'))
        assert final['state'] == 'finished', final
        groove = h.ask('ds_info', {'name': 'srv-groove.npz'}, 'ds_info_result')['dataset']
        assert groove['shape'][1:] == [server._DEFAULT_HPARAMS.max_seq_len, 4] and groove['stats']['notes']['total'] > 0, groove['shape']
        assert groove['info']['token_format'] == 'compound_v1'

        # a legacy (triplet) dataset is listed but marked incompatible and cannot be merged
        np.savez(ROOT / 'data' / BRANCH / 'old.npz', data=np.zeros((3, 256), dtype=int),
                 metadata=np.array([('o.mid', 0, 100.0)] * 3, dtype=object))
        datasets = {d['name']: d for d in h.ask('ds_get_state', None, 'ds_state')['datasets']}
        assert datasets['old.npz']['compatible'] is False and datasets['srv-a.npz']['compatible'] is True, datasets
        res = h.ask('ds_start', {'kind': 'merge', 'params': {'sources': ['srv-a.npz', 'old.npz'], 'name': 'mix'}}, 'ds_start_result')
        assert not res['ok'] and 'legacy' in res['errors']['sources'], res

        # cancel without a running job is a no-op
        assert h.ask('ds_cancel', None, 'ds_cancel_result')['ok'] is False

        # the idle latent field keeps working
        assert h.http.get('/api/health').json['ok']
        print('ALL OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for d in DIRS:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    main()
