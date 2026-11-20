"""CLI entry point for training. The actual logic lives in modules/training.py (shared with the server worker)."""
from hparams import hparams
from modules.train_params import TRAIN_KEYS, MODEL_KEYS
from modules.training import run_training

if __name__ == '__main__':
    # retrain = True
    retrain = False

    # Everything from hparams.py is passed as override so this script behaves like before.
    overrides = {key: getattr(hparams, key) for key in TRAIN_KEYS + MODEL_KEYS}

    if retrain:
        config = {
            'mode': 'resume',
            'resume': {
                'name': 'model_2026-04-27_13-26-37.pt',
                # alternative: 'mode': 'last' / 'best_val' / 'best_train' with 'previous': 0
                'continue_beta': True,        # keep the beta of the checkpoint constant
                'start_lr': 1e-8,             # scheduler warm start
                'end_lr_factor': 0.1,         # scheduler end lr = checkpoint lr * factor
            },
            'overrides': {
                'validation_split': 0,
                'n_epochs': 25,
                'dataset_name': 'dataset-merged-test.npz',
            },
        }
    else:
        config = {'mode': 'new', 'overrides': overrides}

    run_training(config)
