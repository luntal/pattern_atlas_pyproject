d_base = 256

# Compound token format ("compound_v1"): one frame per note = [step, pitch, velocity, deviation].
# Every field has its own index space: 0 = PAD, 1 = SOS, 2 = EOS, raw values start at offset n_specials.
n_specials = 3
n_step = 32       # absolute 16th position inside the 2-bar window (0..31)
n_pitch = 54      # chromatic pitch index (0..53)
n_vel = 32        # quantised velocity (0..31)
n_timing = 33     # micro timing deviation (0..32, center = 16)

field_names = ['step', 'pitch', 'velocity', 'deviation']
field_sizes = [n_step, n_pitch, n_vel, n_timing]
field_vocab_sizes = [size + n_specials for size in field_sizes]


hyper_params = {
    'seed': 42,
    # training
    'dataset_name': 'dataset-groove.npz',
    'validation_split': 0.1,
    'batch_size': 64,
    'n_epochs': 60,
    'learning_rate': 4e-4,
    'scheduler_start_lr': 4e-4,
    'scheduler_end_lr': 4e-5,
    'scheduler_exp_curve_factor': -8.0,
    'scheduler_warmup_ratio': 0.03, # linear LR warm-up (0 -> scheduler_start_lr) over this fraction of all steps; 0 = off. Only used for new trainings.
    'weight_decay': 1e-4,
    
    'n_samples': 0,
    'free_bits': 0.1,
    'global_step': 0,
    'global_epoch': 0, # num of trained epochs
    'annealing_steps_norm': 1, # normalized amount of steps. 1 equals number of steps in all epochs, 0.5 equals number of steps in half of the epochs, etc.
    'annealing_list': 
    [[0.0, 0.0],
     [0.1, 0.1],
     [0.4, 0.6],
     [1.0, 0.6],
     ], # list of [normalized_position, beta_value] pairs. Normalized step is between 0 and 1, where 1 is the last training step. Beta value is the annealed beta at that point; segments are exponentially smoothed between points.
    'annealing_slope': 0.05, # slope of exponential smoothing between annealing points. 0 means linear, higher values mean more step-like changes.
    'weighted_beta': 0.0, # current beta curing training.

    # checkpoints / stopping
    'last_checkpoint_interval': 3, # write model_last_*.pt every N epochs (and at the end / on stop)
    'early_stopping': False, # stop when the validation loss (train loss without validation) no longer improves
    'early_stopping_patience': 8, # epochs without improvement before stopping
    'early_stopping_min_delta': 0.001, # minimal loss decrease that counts as improvement

    # model
    'latent_dim': d_base * 2,
    'n_memory_tokens': 32,

    'steps_per_bar': 32,
    'height': n_pitch,
    'width': n_step,
    'channels': 1,

    'd_model': d_base, # embedding dim and model dimension
    'n_heads': 4,
    'n_layers': 4,
    'ff_dim': d_base * 2,
    'dropout': 0.1,
    'norm_first': True, # pre-norm transformer layers (stable at higher LR); False = post-norm like older checkpoints
    'max_seq_len': 128, # max number of frames (notes + EOS) per 2-bar window

    # compound tokens (frames):
    'token_format': 'compound_v1',
    'n_specials': n_specials,
    'pad_token': 0,
    'sos_token': 1,
    'eos_token': 2,
    'field_names': field_names,
    'field_sizes': field_sizes,
    'field_vocab_sizes': field_vocab_sizes,
    'field_loss_weights': [1.0, 1.0, 1.0, 1.0], # weights of the per field cross entropies (step, pitch, velocity, deviation)
    'n_step': n_step,
    'n_pitch': n_pitch,
    'n_vel': n_vel,
    'n_timing': n_timing,
    'center_timing': n_timing // 2,
}
print_info = False
# print_info = True
if print_info:
    print('token format:', hyper_params['token_format'])
    print('fields:', field_names)
    print('field vocab sizes:', field_vocab_sizes)
    print('pad / sos / eos:', hyper_params['pad_token'], hyper_params['sos_token'], hyper_params['eos_token'])
    print('center timing (raw):', hyper_params['center_timing'])

from types import SimpleNamespace

hparams = SimpleNamespace(**hyper_params)