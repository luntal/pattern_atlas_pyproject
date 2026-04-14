import numpy as np
import mido
import time
import os
from pathlib import Path
from modules.branch_paths import latest_in_branch, safe_branch_name, get_current_branch
from modules.model import TransformerVAE, reparameterize, ONNXWrapper
from types import SimpleNamespace
import torch
import math
import glob

def load_model(previous=0, onnx=False ):
    # Find models in the current branch-specific models directory.
    branch_name = safe_branch_name(get_current_branch())
    model_dir = Path('models') / branch_name
    model_files = sorted(model_dir.glob('model_*.pt'), key=os.path.getmtime)

    if not model_files:
        raise FileNotFoundError(f"No model files found in models/{branch_name}/")

    previous = max(int(previous), 0)
    selected_index = len(model_files) - 1 - previous
    if selected_index < 0:
        selected_index = 0
        print(
            f"Requested previous={previous}, but only {len(model_files)} model(s) are available in models/{branch_name}/. "
            f"Loading oldest model instead: {model_files[selected_index]}"
        )
        previous = len(model_files) - 1

    selected_model = model_files[selected_index]
    if previous == 0:
        print(f'Loading latest model: {selected_model}')
    else:
        print(f'Loading model with previous={previous}: {selected_model}')

    # Load the saved data
    checkpoint = torch.load(str(selected_model), map_location='cpu')
    model_state_dict = checkpoint['model_state_dict']
    hparams_dict = checkpoint['hparams']

    # Recreate the hparams object
    hparams = SimpleNamespace(**hparams_dict)

    if onnx:
        model = ONNXWrapper(hparams)
    else:
        model = TransformerVAE(hparams)
        model.load_state_dict(model_state_dict)
        print('--' * 10)
        print('model config: ')
        print('model dimensions: ', hparams.d_model)
        print('n_layers: ', hparams.n_layers)
        print('n_heads: ', hparams.n_heads)
        print('--' * 10)
    model.eval()
    return model, hparams

def load_dataset(hparams, project_root):
    # dataset_name = hparams.dataset_name
    branch_name = safe_branch_name(get_current_branch())
    dataset_name = os.path.join(project_root, 'data', branch_name, hparams.dataset_name)
    print(f'Loading dataset: {dataset_name}')
    data = np.load(dataset_name)['data']
    print('Total amount of samples: ', data.shape)
    if hparams.n_samples < data.shape[0] and hparams.n_samples > 0:
        data = data[:hparams.n_samples]
    elif hparams.n_samples < 0:
        pass
    elif hparams.n_samples == 0:
        hparams.n_samples = data.shape[0]
        
    
    print('Using ', data.shape[0])
    print('---'*5)

    # Shuffle data before splitting
    np.random.seed(hparams.seed)
    np.random.shuffle(data)
    return data

def load_scheduler(optim, loader, hparams):
        start_lr = getattr(hparams, 'scheduler_start_lr', hparams.learning_rate)
        end_lr = getattr(hparams, 'scheduler_end_lr', hparams.learning_rate * 0.1)
        exp_curve_factor = getattr(hparams, 'scheduler_exp_curve_factor', 1.0)
        total_scheduler_steps = max(len(loader) * hparams.n_epochs, 1)

        for param_group in optim.param_groups:
            param_group['lr'] = start_lr

        def lr_lambda(current_step):
            if total_scheduler_steps <= 1:
                return end_lr / start_lr

            progress = min(max(current_step / (total_scheduler_steps - 1), 0.0), 1.0)

            if math.isclose(exp_curve_factor, 0.0, abs_tol=1e-12):
                curved_progress = progress
            else:
                numerator = math.exp(exp_curve_factor * progress) - 1.0
                denominator = math.exp(exp_curve_factor) - 1.0
                curved_progress = numerator / denominator

            current_lr = start_lr + (end_lr - start_lr) * curved_progress
            return current_lr / start_lr

        return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

def play_midi(_global_vars):
    # global playback_position
    # global playback_cursor_matrix
    midi_output_port_name = mido.get_output_names()[0]
    print('using midi port ', midi_output_port_name)
    output_port = mido.open_output(midi_output_port_name, callback=lambda msg: print(msg))

    vel = 0
    while not _global_vars.stop_event.is_set():
        playback_position = _global_vars.get_step()
        _current_index = _global_vars.get_current_index()
        matrix = _global_vars.get_matrices()[_current_index]
        matrix = np.array(matrix)
        # print(playback_position)

        for pitch in range(_global_vars.height):
            if matrix[pitch, playback_position] > 0:
                vel = int(matrix[pitch, playback_position] * 127)
                # print(pitch + 36)
                output_port.send(mido.Message('note_on', note=pitch + 36, velocity=vel))
        time.sleep(_global_vars.sixteenth_note_length)
        for i in range(_global_vars.height):
            output_port.send(mido.Message('note_off', note=i + 36, velocity=vel))
        # playback_cursor_matrix = np.roll(playback_cursor_matrix, -1, axis=1)
        # playback_position += 1
        # playback_position %= hparams.width
        _global_vars.step()

def seq2mat(seq, _hparams):
    mat = np.zeros((_hparams.height, _hparams.width))
    idx = 0
    vel = 0
    for token in seq:
        if _hparams.min_vel_token <= token < _hparams.max_vel_token:
            vel = token - _hparams.min_vel_token
            vel /= (_hparams.n_vel - 1)
        if _hparams.min_pitch_token <= token < _hparams.max_pitch_token:
            pitch = token - _hparams.min_pitch_token
            mat[pitch, idx] = vel
        if token == _hparams.step_token:
            idx += 1
        if idx >= _hparams.width:
            break
    return mat


def syncopation_measure(mat: np.ndarray) -> float:
    '''
    takes a matrix and calculates the syncopation measure for it
    :param mat:
    :return:
    '''
    # global hparams
    syncopation_measure = 0
    # syncopation_weights = [0, 4, 3, 4, 1, 4, 3, 4, 1, 4, 3, 4, 1, 4, 3, 4] * 2
    syncopation_weights = [-1, 2, 1, 2] * 4 * 2
    n_onsets = 0
    sum_weights = 0
    full = 0

    # mat = seq2mat(sequence, hparams)
    for pos in range(mat.shape[1]):
        vel_sum = 0  # sum of velocities at current position
        c = 0
        n_onsets_at_pos = 0
        for pitch in range(mat.shape[0]):
            vel = mat[pitch, pos]
            if vel > 0:
                vel_sum += vel
                n_onsets += 1
                c += 1
                sum_weights += syncopation_weights[pos]
                n_onsets_at_pos += 1
        # vel_sum /= c
        syncopation_measure += syncopation_weights[pos] * vel_sum
        full += (syncopation_weights[pos] * n_onsets_at_pos)
    if sum_weights > 0:
        syncopation_measure /= sum_weights
        full /= sum_weights
        # print(full)
    else:
        syncopation_measure = 0
    return min(1, max(0, syncopation_measure))

def syncopation_measure_weighted(mat: np.ndarray, max_simultaneous_onsets: int=3) -> float:
    syncopation_measure = 0
    syncopation_weights = [0, 2, 1, 2] * 4 * 2

    instrument_weights = [1, 1, 1, 1, 1, 1, 0.1, 1, 0.1, 1, 0.1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    max_velos = mat.max(axis=0)
    vel_thr = max_velos * 0.75
    for pitch in range(mat.shape[0]):
        n_onsets = 0
        syncopation_measure_single_pitch = 0
        for pos in range(mat.shape[1]):
            vel = mat[pitch, pos]
            if vel > vel_thr[pitch]:
                n_onsets += 1
                syncopation_measure_single_pitch += syncopation_weights[pos] * vel
        if n_onsets == 0: continue
        syncopation_measure_single_pitch /= n_onsets
        syncopation_measure += (instrument_weights[pitch] * syncopation_measure_single_pitch)
    syncopation_measure /= max_simultaneous_onsets
    syncopation_measure = min(1, max(0, syncopation_measure))
    return syncopation_measure



def vel_deviation_measure(mat: np.ndarray):
    '''
    takes a matrix and calculates the velocity deviation measure for it
    :param mat:
    :return:
    '''
    vel_deviation = 0
    n_onsets = 0
    for pos in range(mat.shape[1]):
        prev_vel = 0
        for pitch in range(mat.shape[0]):
            vel = mat[pitch, pos]
            if vel > 0:
                n_onsets += 1
                if vel != prev_vel:
                    vel_deviation += abs(vel - prev_vel)
                prev_vel = vel
    return vel_deviation / n_onsets

def note_density(mat: np.ndarray, max_notes: int = 4):
    if max_notes > mat.shape[0]: max_notes = mat.shape[0]
    density = np.sum(np.where(mat != 0, 1, 0)) / (max_notes * mat.shape[1])
    if density > 1: density = 1
    # print(density)
    return density



