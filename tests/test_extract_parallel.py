"""Parallel MIDI extraction (process pool) in MidiProcessor.extract.

Checks: same result as the old sequential implementation, same result for 1 and several workers (also with the random
parts switched on), per-file seeding, unchanged global RNG state, progress callback, cancel, and the automatic
worker choice. Uses the files in midifiles/.

Run from the project root:  python tests/test_extract_parallel.py
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

BRANCH = '_test_extract_parallel'
os.environ['GITHUB_REF_NAME'] = BRANCH  # throw-away data/<branch> folder for the dataset job
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np  # noqa: E402

from hparams import hparams as HP  # noqa: E402
from modules import midi_processor as mp  # noqa: E402

MIDI = str(ROOT / 'midifiles')
COMMON = dict(midi_folder_path=MIDI, drums_only=False, filter_duplicates=False, jump=1,
              groove_to_specific=True, clip=False)


def run(processor_cls=mp.MidiProcessor, **kwargs):
    processor = processor_cls(SimpleNamespace(**vars(HP)))
    args = dict(COMMON)
    args.update(kwargs)
    processor.extract(**args)
    return processor


def same(a, b):
    assert len(a.sequences) == len(b.sequences), (len(a.sequences), len(b.sequences))
    for x, y in zip(a.sequences, b.sequences):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    assert [tuple(m) for m in a.metadata] == [tuple(m) for m in b.metadata]


def load_old_module():
    """The extraction as it was in git HEAD (before the parallelization)."""
    code = subprocess.check_output(['git', '-C', str(ROOT), 'show', 'HEAD:src/modules/midi_processor.py'], text=True)
    tmp = Path(tempfile.mkdtemp()) / 'midi_processor_old.py'
    tmp.write_text(code, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('midi_processor_old', tmp)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    n_files = len(list(Path(MIDI).rglob('*.mid*')))
    assert n_files >= mp.PARALLEL_MIN_FILES, f'need at least {mp.PARALLEL_MIN_FILES} midi files, found {n_files}'

    # worker choice
    assert mp.MidiProcessor._resolve_workers(1, 100) == 1
    assert mp.MidiProcessor._resolve_workers(4, 100) == 4
    assert mp.MidiProcessor._resolve_workers(4, 3) == 1            # too few files: sequential
    assert mp.MidiProcessor._resolve_workers(64, 20) == 20
    assert 1 <= mp.MidiProcessor._resolve_workers(None, 100) <= 8
    assert mp.MidiProcessor._resolve_workers(0, 100) >= 1

    # 1) deterministic settings: identical to the old sequential implementation
    t0 = time.time()
    seq = run(workers=1)
    t_seq = time.time() - t0
    assert len(seq.sequences) > 0
    try:
        old = load_old_module()
        old_proc = run(processor_cls=old.MidiProcessor, jump=1)
        same(seq, old_proc)
        print(f'OK identical to old sequential code ({len(seq.sequences)} sequences)')
    except subprocess.CalledProcessError:
        print('skip: git HEAD version not available')

    # 2) pool gives the same result in file order
    t0 = time.time()
    par = run(workers=3)
    t_par = time.time() - t0
    same(seq, par)
    assert par.last_stats.max_notes_per_window == seq.last_stats.max_notes_per_window
    for key in ('skipped_invalid', 'skipped_time_signature', 'skipped_too_short', 'one_bar_patterns',
                'truncated_sequences', 'merged_duplicates', 'skipped_no_drums'):
        assert getattr(par.last_stats, key) == getattr(seq.last_stats, key), key
    print(f'OK pool == sequential (1 worker {t_seq:.1f}s, 3 workers {t_par:.1f}s)')

    # 3) random parts: reproducible with a seed, independent of the number of workers, different per seed
    rnd = dict(randomize_velocity=0.3, qnt_strength=1.0)
    a = run(workers=1, seed=7, **rnd)
    b = run(workers=3, seed=7, **rnd)
    c = run(workers=3, seed=8, **rnd)
    same(a, b)
    assert not all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a.sequences, c.sequences)), \
        'different seeds must give different random parts'
    assert a.last_seed == 7
    print('OK seeded randomness: workers=1 == workers=3, other seed differs')

    # 4) global RNG state is left alone
    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)
    run(workers=1, seed=1, **rnd)
    assert np.random.rand() == expected, 'extract must not disturb the global numpy RNG'
    print('OK global RNG untouched')

    # 5) progress callback: file_idx grows, final call has file_idx == n_files, stats are filled in
    calls = []
    run(workers=3, progress_callback=lambda i, n, name, stats, n_seq: calls.append((i, n, n_seq)))
    assert [c[0] for c in calls] == list(range(len(calls)))
    assert calls[-1][0] == calls[-1][1] and calls[-1][2] == len(seq.sequences)
    assert all(calls[i][2] <= calls[i + 1][2] for i in range(len(calls) - 1))
    print(f'OK progress ({len(calls)} callbacks)')

    # 6) cancel while the pool is running
    state = {'n': 0}

    def stop_after_some():
        state['n'] += 1
        return state['n'] > 3

    t0 = time.time()
    try:
        run(workers=3, should_stop=stop_after_some)
        raise AssertionError('ExtractionCancelled expected')
    except mp.ExtractionCancelled:
        pass
    print(f'OK cancel with pool ({time.time() - t0:.1f}s)')

    # 7) small job stays sequential and still works
    small = run(take=3, workers=4)
    assert len(small.sequences) > 0

    # 8) Dataset-tab job (extraction runs in a thread of the server process and starts the pool from there)
    from modules import dataset_tools as dt
    params, errors = dt.validate_extract({'folder': MIDI, 'name': 'x', 'workers': 99})
    assert 'workers' in errors, errors
    params, errors = dt.validate_extract({'folder': MIDI, 'name': 'x', 'workers': '3', 'seed': '5'})
    assert not errors and params['workers'] == 3 and params['seed'] == 5, (params, errors)
    params, errors = dt.validate_extract({'folder': MIDI, 'name': 'x', 'seed': ''})
    assert not errors and params['seed'] is None and params['workers'] == 0

    data_dir = dt.data_dir()
    try:
        manager = dt.DatasetJobManager()
        ok, errors = manager.start('extract', {'folder': MIDI, 'name': 'pool-job', 'mapping': 'groove_to_specific',
                                               'workers': 3, 'seed': 11, 'jump': 1, 'randomize_velocity': 0.2})
        assert ok, errors
        t0 = time.time()
        while manager.is_active():
            assert time.time() - t0 < 300, 'job timed out'
            time.sleep(0.2)
        snapshot = manager.snapshot()
        assert snapshot['state'] == 'finished', snapshot
        info = dt.dataset_info('pool-job.npz')
        assert info['info']['options']['seed'] == 11 and info['info']['options']['workers'] == 3, info['info']['options']
        with np.load(data_dir / 'pool-job.npz', allow_pickle=True) as npz:
            assert npz['data'].shape[0] > 0 and npz['data'].shape[2] == 4
        print(f"OK dataset job with pool ({snapshot['stats']['sequences']} sequences)")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
    print('ALL OK')


if __name__ == '__main__':
    main()
