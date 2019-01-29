import os
import random

import numpy as np
import scipy as sp
from librosa.effects import pitch_shift
from scipy.io import wavfile


def read_wavfile(filepath):
    assert os.path.exists(filepath), filepath + "not found!"
    sample_rate, wave = wavfile.read(filepath)
    wave = wave.astype(np.float32)
    return sample_rate, wave


def normalize_wavfile(wav, mode=2, normalize_to_peak=False):
    # Normalizes the amplitude of the wav file to belong to the [-1, 1] interval
    # param mode: 0 is 32-bit floating point, 1 is 32-bit PCM, 2 is 16-bit pcm (default)
    # and 3 is 8-bit pcm
    if normalize_to_peak:
        peak = np.max(np.abs(wav))
        if peak == 0:
            raise ValueError("wavfile is constant zero, impossible to normalize!")
        elif np.isnan(wav).any():
            raise ValueError("wavfile contains nan values, corrupted!")
        else:
            return wav / peak
    if mode == 0:
        return wav
    elif mode == 1:
        return wav/2147483648
    elif mode == 2:
        return wav/32768
    elif mode == 3:
        return 2*(wav/255-0.5)
    else:
        raise ValueError(f"Mode {mode} not recognized as a valid mode. Use one of the following: [0, 1, 2, 3]")


def fix_wavfile_duration(wav, sample_rate, duration):
    desired_length = int(duration*sample_rate)
    if desired_length == len(wav):  # Correct length
        return wav
    else:
        return fix_wavfile_length(wav, desired_length)


def fix_wavfile_length(wav, length):
    if length == len(wav):
        return wav
    elif length > len(wav): # Zero-pad
        samples_to_pad = length - len(wav)
        n_left = int(samples_to_pad / 2)
        n_right = (samples_to_pad - n_left)
        return np.concatenate([[0] * n_left, wav, [0] * n_right])
    else: # Center-crop
        samples_to_cut = len(wav) - length
        left_idx = int(samples_to_cut / 2)
        right_idx = len(wav) - (samples_to_cut - left_idx)
        return wav[left_idx:right_idx]


def saturate_wavfile(wav, prop_amplitude_cut):
    if prop_amplitude_cut < 1:
        peak = max(abs(max(wav)), abs(min(wav)))
        limit = peak * prop_amplitude_cut
        return np.clip(wav, -limit, limit) * peak/limit
    else:
        return wav / prop_amplitude_cut


def resample_wavfile(wav, factor):
    original_length = len(wav)
    wav = sp.signal.resample(wav, int(len(wav) * factor))
    wav = fix_wavfile_length(wav, original_length)
    return wav


def time_offset_wavfile(wav, shift_factor):
    n_samples_shift = int(abs(shift_factor) * len(wav))
    if shift_factor > 0:  # shift to the right (cut rightmost samples)
        return np.concatenate([[0] * n_samples_shift, wav[:-n_samples_shift]])
    else:
        return np.concatenate([wav[n_samples_shift:], [0] * n_samples_shift])


def add_noise_to_wavfile(wav, amplitude_factor, clip_to_original_range=False):
    max_wav = max(wav)
    min_wav = min(wav)
    noise = np.random.rand(*wav.shape) * (max_wav - min_wav) - min_wav
    wav = wav + amplitude_factor * noise
    if clip_to_original_range:
        wav = np.clip(wav, min_wav, max_wav)
    return wav


def pitch_shift_wavfile(wav, sr, n_octaves):
    peak = max(np.abs(np.max(wav)), np.abs(np.min(wav)))
    if n_octaves == 0:
        return wav
    new_wav = pitch_shift(y=wav.astype(float), sr=sr, n_steps=n_octaves * 12, bins_per_octave=12)
    new_peak = max(np.abs(np.max(new_wav)), np.abs(np.min(new_wav)))
    new_wav = peak * new_wav / new_peak
    return new_wav


def randomly_distort_wavfile(wav, sr=16000, pitch=True, resample=True, saturate=True, offset=True, noise=True):
    peak = max(np.abs(np.max(wav)), np.abs(np.min(wav)))
    if pitch:
        rand_pitch_shift = random.normalvariate(0, 0.1)
        wav = pitch_shift_wavfile(wav, sr, rand_pitch_shift)
    if resample:
        rand_resample = random.normalvariate(0, 0.2)
        wav = resample_wavfile(wav, 1 / (1 + abs(rand_resample)) if rand_resample < 0 else (1 + rand_resample))
    if saturate:
        wav = saturate_wavfile(wav, 1.2 / (1 + abs(random.expovariate(2.5))))
    if offset:
        wav = time_offset_wavfile(wav, random.normalvariate(0, 0.1))
    if noise:
        wav = add_noise_to_wavfile(wav, random.normalvariate(0, 0.05) * (random.random() > 0.7), True)
    new_peak = max(np.abs(np.max(wav)), np.abs(np.min(wav)))
    wav = peak * wav / new_peak
    return wav


def draw_random_subclip(clip, n_samples):
    assert (len(clip) > n_samples)
    index = random.choice(range(len(clip) - 1 - n_samples))
    return clip[index:(index + n_samples)]
