"""Parse cache and vectorized note preparation of MidiProcessor.extract.

Checks that the result does not depend on the cache (cold / warm / disabled / corrupt entry / changed file) or on
the worker pool, and that the array core gives the same frames as the dict based API.

Run:  python tests/test_extract_cache.py   (prints "ALL OK")
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MIDI = ROOT / 'midifiles'
sys.path.insert(0, str(ROOT / 'src'))


def make_processor():
    from hparams import hparams
    from modules import midi_processor as mp
    return mp.MidiProcessor(SimpleNamespace(**vars(hparams)))


def run(midi_dir, **kwargs):
    processor = make_processor()
    options = dict(midi_folder_path=str(midi_dir), workers=1, seed=3, groove_to_specific=True, jump=1)
    options.update(kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        t0 = time.time()
        processor.extract(**options)
    return processor, time.time() - t0


def same(a, b):
    return (
        len(a.sequences) == len(b.sequences)
        and all(np.array_equal(x, y) for x, y in zip(a.sequences, b.sequences))
        and [tuple(map(str, m)) for m in a.metadata] == [tuple(map(str, m)) for m in b.metadata]
        and asdict(a.last_stats) == asdict(b.last_stats)
    )


def main():
    cache_dir = Path(tempfile.mkdtemp(prefix='midi_cache_test_'))
    src_dir = Path(tempfile.mkdtemp(prefix='midi_src_'))
    os.environ['RBF_MIDI_CACHE_DIR'] = str(cache_dir)  # inherited by the pool workers
    os.environ.pop('RBF_MIDI_CACHE', None)
    try:
        from modules import midi_processor as mp

        files = sorted(MIDI.rglob('*.mid*'))
        assert len(files) >= mp.PARALLEL_MIN_FILES, f'need at least {mp.PARALLEL_MIN_FILES} midi files'
        for f in files:
            shutil.copy2(f, src_dir / f.name)

        def cache_files():
            return list(cache_dir.rglob('*.npz'))

        cases = [
            dict(),
            dict(jump=2, drums_only=True, groove_to_specific=False),
            dict(generic_midi_mapping=True, groove_to_specific=False),
            dict(randomize_velocity=0.2, clip=True, filter_duplicates=False),
            dict(generic_to_chromatic=True, groove_to_specific=False),
        ]
        for i, kw in enumerate(cases):
            plain, _ = run(src_dir, parse_cache=False, **kw)
            assert not cache_files(), 'parse_cache=False must not write cache entries'
            cold, t_cold = run(src_dir, **kw)
            n_cold = len(cache_files())
            warm, t_warm = run(src_dir, **kw)
            assert n_cold > 0 and len(cache_files()) == n_cold
            assert same(plain, cold) and same(plain, warm), f'case {i}: cache changed the result'
            print(f'case {i}: {len(plain.sequences)} sequences, cache entries {n_cold}, '
                  f'cold {t_cold:.2f}s, warm {t_warm:.2f}s: identical')
            shutil.rmtree(cache_dir, ignore_errors=True)

        # environment switch disables the cache
        os.environ['RBF_MIDI_CACHE'] = '0'
        run(src_dir)
        assert not cache_files(), 'RBF_MIDI_CACHE=0 must not write cache entries'
        os.environ.pop('RBF_MIDI_CACHE')

        # corrupt and outdated entries are re-parsed (and rewritten)
        reference, _ = run(src_dir, parse_cache=False)
        run(src_dir)
        entries = cache_files()
        entries[0].write_bytes(b'garbage')
        again, _ = run(src_dir)
        assert same(reference, again), 'corrupt cache entry changed the result'
        assert entries[0].stat().st_size > 100, 'corrupt cache entry was not rewritten'

        # a changed file (mtime) gets its own entry, the result stays correct
        before = len(cache_files())
        victim = sorted(src_dir.glob('*.mid*'))[0]
        future = time.time() + 1000
        os.utime(victim, (future, future))
        changed, _ = run(src_dir)
        assert len(cache_files()) == before + 1
        assert same(reference, changed)

        # process pool with a warm cache
        pooled, _ = run(src_dir, workers=3)
        assert same(reference, pooled), 'pool + cache differs from sequential'

        # array core == dict based API
        processor = make_processor()
        rng = np.random.default_rng(0)
        n = 60
        c_window = np.sort(rng.integers(0, 6, n))
        step = rng.integers(0, 32, n)
        pitch = rng.integers(0, 54, n)
        vel = rng.integers(0, 32, n)
        dev = rng.integers(0, 33, n)
        notes = [dict(c_window=int(c_window[i]), c_step_in_window=int(step[i]), pitch_idx=int(pitch[i]),
                      velocity_idx=int(vel[i]), dev_idx=int(dev[i]), filename='x.mid', bpm=120.0, bpm_from_name=None)
                 for i in range(n)]
        for jump in (1, 2, 3):
            a_seq, a_meta = processor.notes_advanced_to_token_sequences_stream(notes, 32, 2, max_window_exclusive=5, jump=jump)
            b_seq, b_meta = processor._frames_from_note_arrays('x.mid', 120.0, c_window, step, pitch, vel, dev, 32, 2,
                                                              max_window_exclusive=5, jump=jump)
            assert len(a_seq) == len(b_seq) > 0 and a_meta == b_meta
            assert all(np.array_equal(x, y) for x, y in zip(a_seq, b_seq))
        print('array core equals dict API')
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.rmtree(src_dir, ignore_errors=True)
    print('ALL OK')


if __name__ == '__main__':
    main()
