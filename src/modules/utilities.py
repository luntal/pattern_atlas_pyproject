import numpy as np
import mido
import time


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



