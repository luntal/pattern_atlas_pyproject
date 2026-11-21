"""Tests for modules/hparams_io.save_hparams on a copy of src/hparams.py.

Run from the project root:  python tests/test_hparams_io.py
"""
import importlib.util
import os
import shutil
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from modules.hparams_io import HparamsSaveError, save_hparams  # noqa: E402

SOURCE = os.path.join(ROOT, 'src', 'hparams.py')


def load_values(path):
    spec = importlib.util.spec_from_file_location('hparams_copy_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.hyper_params, module.hparams


def make_copy():
    tmp = tempfile.mkdtemp(prefix='hparams_io_')
    path = os.path.join(tmp, 'hparams.py')
    shutil.copy(SOURCE, path)
    return tmp, path


def test_value_changes():
    tmp, path = make_copy()
    try:
        before, _ = load_values(path)
        original_text = open(path, encoding='utf-8').read()
        new_anneal = [[0.0, 0.0], [0.2, 0.05], [0.5, 0.3], [1.0, 0.3]]
        result = save_hparams({
            'learning_rate': 1e-3, 'n_epochs': 10, 'annealing_list': new_anneal, 'latent_dim': 300,
            'dataset_name': 'other.npz', 'free_bits': 0.1,  # free_bits is unchanged
        }, path=path)
        after, ns = load_values(path)
        assert after['learning_rate'] == 1e-3 and after['n_epochs'] == 10
        assert after['annealing_list'] == new_anneal
        assert after['latent_dim'] == 300 and after['dataset_name'] == 'other.npz'
        assert 'free_bits' in result['unchanged'] and 'free_bits' not in result['changed']
        assert ns.latent_dim == 300
        # everything else is identical
        for key, value in before.items():
            if key not in result['changed']:
                assert after[key] == value, key
        # comments and structure are kept
        new_text = open(path, encoding='utf-8').read()
        for marker in ["# embedding dim and model dimension", "# num of trained epochs",
                       "# list of [normalized_position, beta_value] pairs", "# slope of exponential smoothing",
                       "# training", "# compound tokens (frames):", "'ff_dim': d_base * 2,"]:
            assert marker in new_text, marker
        assert os.path.exists(path + '.bak')
        assert open(path + '.bak', encoding='utf-8').read() == original_text
        print('ok value changes')
    finally:
        shutil.rmtree(tmp)


def test_noop_keeps_file_identical():
    tmp, path = make_copy()
    try:
        before = open(path, 'rb').read()
        before_values, _ = load_values(path)
        result = save_hparams({k: before_values[k] for k in ('learning_rate', 'n_epochs', 'annealing_list', 'd_model',
                                                              'latent_dim', 'ff_dim', 'dataset_name')}, path=path)
        assert result['changed'] == {} and result['backup'] is None
        assert open(path, 'rb').read() == before
        print('ok no-op leaves file byte-identical')
    finally:
        shutil.rmtree(tmp)


def test_d_model_updates_d_base():
    tmp, path = make_copy()
    try:
        result = save_hparams({'d_model': 128, 'n_heads': 4}, path=path)
        after, _ = load_values(path)
        assert after['d_model'] == 128
        assert after['latent_dim'] == 256 and after['ff_dim'] == 256, 'derived values follow d_base'
        assert 'd_base = 128' in open(path, encoding='utf-8').read()
        assert result['derived'].get('latent_dim') == 256
        # explicit derived values that differ are written as literals
        save_hparams({'d_model': 64, 'latent_dim': 512}, path=path)
        after, _ = load_values(path)
        assert after['d_model'] == 64 and after['latent_dim'] == 512 and after['ff_dim'] == 128
        print('ok d_model -> d_base with derived values')
    finally:
        shutil.rmtree(tmp)


def test_rejections():
    tmp, path = make_copy()
    try:
        before = open(path, 'rb').read()
        for changes in ({'pad_token': 5}, {'vocab_size': 10}, {'ranges': {}}, {'seed': 'abc'},
                        {'batch_size': 1.5}, {'annealing_list': [[0.0]]}, {'d_model': 30, 'n_heads': 4},
                        {'learning_rate': float('nan')}):
            try:
                save_hparams(changes, path=path)
            except HparamsSaveError:
                assert open(path, 'rb').read() == before, f'file changed for {changes}'
                continue
            raise AssertionError(f'expected rejection for {changes}')
        assert not os.path.exists(path + '.bak')
        print('ok rejected invalid changes, file untouched')
    finally:
        shutil.rmtree(tmp)


if __name__ == '__main__':
    test_value_changes()
    test_noop_keeps_file_identical()
    test_d_model_updates_d_base()
    test_rejections()
    print('ALL OK')
