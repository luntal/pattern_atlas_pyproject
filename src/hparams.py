d_base = 256

ranges = {
    'specials': 3,
    'step': 1,
    'pitch': 24,
    'velocity': 32,
    'timing': 33,
}

vocab_size = sum(ranges.values())

specials_index = list(ranges.keys()).index('specials')
step_index = list(ranges.keys()).index('step')
pitch_index = list(ranges.keys()).index('pitch')
velocity_index = list(ranges.keys()).index('velocity')
timing_index = list(ranges.keys()).index('timing')

keys = list(ranges.keys())

min_tokens = []
current_min_token = 0
for i in range(len(keys)):
    min_tokens.append(current_min_token)
    current_min_token += ranges[keys[i]]
# print(min_tokens)
# print(min_tokens[step_index])


hyper_params = {
    # training
    'dataset_name': 'dataset-merged.npz',
    'validation_split': 0.1,
    'batch_size': 32,
    'n_epochs': 50,
    'learning_rate': 3e-4,
    'n_samples': 0,
    # 'annealing_steps': 250 * 200, # kl annealing should start after around 50% of epochs, is computed in train method
    'beta': 0.3,
    'free_bits': 0.05,
    'global_step': 0,
    'global_epoch': 0, # num of trained epochs
    'kl_annealing_step': 0,
    'current_lr': 0.0,

    'latent_dim': d_base // 2,

    'steps_per_bar': 32,
    'total_steps': 32 * 2,
    'height': 24,
    'width': 32,
    'channels': 1,
    'vocab_size': vocab_size,

    'd_model': d_base, # embedding dim and model dimension
    'n_heads': 4,
    'n_layers': 4,
    'ff_dim': d_base * 2,
    'dropout': 0.1,
    'max_seq_len': 256,

    # tokens:
    'min_specials_token': min_tokens[specials_index],
    'max_specials_token': min_tokens[specials_index] + ranges['specials'],
    'pad_token': min_tokens[specials_index],
    'sos_token': min_tokens[specials_index] + 1,
    'eos_token': min_tokens[specials_index] + 2,
    'n_steps': ranges['step'],
    'step_token': min_tokens[step_index],
    # 'sixteenth_step_token': min_tokens[step_index],
    'n_pitch': ranges['pitch'],
    'min_pitch_token': min_tokens[pitch_index],
    'max_pitch_token': min_tokens[pitch_index] + ranges['pitch'] ,
    'n_vel': ranges['velocity'],
    'min_vel_token': min_tokens[velocity_index],
    'max_vel_token': min_tokens[velocity_index] + ranges['velocity'],
    'n_timing': ranges['timing'],
    'min_timing_token': min_tokens[timing_index],
    'max_timing_token': min_tokens[timing_index] + ranges['timing'],

    'token_bias': 4,
}
print_info = False
# print_info = True
if print_info:
    print('total number of tokens: ', vocab_size)

    print('pad token:', hyper_params['pad_token'])
    print('sos token:', hyper_params['sos_token'])
    print('eos token:', hyper_params['eos_token'])
    print('step token:', hyper_params['step_token']),
    print('pitch range: ', hyper_params['min_pitch_token'], hyper_params['max_pitch_token'])
    print('velocity range', hyper_params['min_vel_token'], hyper_params['max_vel_token'])
    print('timing range', hyper_params['min_timing_token'], hyper_params['max_timing_token'])
    print('Ranges are excluding upper bound!')

from types import SimpleNamespace

hparams = SimpleNamespace(**hyper_params)