import os
import random
from multiprocessing.dummy import Pool

import numpy as np
from scipy.io import wavfile

from src.common_paths import get_training_data_path
from src.data_tools import read_wavfile, draw_random_subclip, randomly_distort_wavfile, fix_wavfile_length, \
    normalize_wavfile


def get_list_of_wav_paths(include_augmentations=False):
    folders = ("audio",) if not include_augmentations else ("audio", "augmented")
    list_test = open(os.path.join(get_training_data_path(), "testing_list.txt"))
    list_test = list(map(lambda x: os.path.normpath(os.path.join(get_training_data_path(), "audio", x.strip())),
                         list_test))
    list_val = open(os.path.join(get_training_data_path(), "validation_list.txt"))
    list_val = list(map(lambda x: os.path.normpath(os.path.join(get_training_data_path(), "audio", x.strip())),
                        list_val))
    list_train = list()
    for folder in folders:
        base_path = os.path.join(get_training_data_path(), folder)
        for dirName, subdirList, fileList in os.walk(base_path):
            for fname in fileList:
                if fname.endswith("wav"):
                    filepath = os.path.join(base_path, os.path.split(dirName)[-1], fname)
                    list_train.append(os.path.normpath(filepath.replace(os.sep, "/")).strip())
    list_train = np.setdiff1d(list_train, list_test)
    list_train = np.setdiff1d(list_train, list_val)
    list_train = list_train.tolist()
    list_train = list(filter(lambda x: "background_noise" not in x, list_train))
    return list_train, list_val, list_test


def generate_white_noise_clip(n_samples):
    clip = np.random.randn(n_samples)
    clip /= np.abs(clip).max()
    return clip


def load_real_noise_clips():
    clips = []
    path = os.path.join(get_training_data_path(), "audio", "_background_noise_")
    for filename in filter(lambda x: x.endswith(".wav"), os.listdir(path)):
        _, wav = read_wavfile(os.path.join(path, filename))
        clips.append(wav)
    return clips


def load_random_real_noise_clip():
    clips = []
    path = os.path.join(get_training_data_path(), "audio", "_background_noise_")
    filename = random.choice(list(filter(lambda x: x.endswith(".wav"), os.listdir(path))))
    _, clip = read_wavfile(os.path.join(path, filename))
    return clip


def get_random_real_noise_subclip(n_samples, noise_clips=None):
    if noise_clips is None:
        noise_clip = load_random_real_noise_clip()
    else:
        noise_clip = random.choice(noise_clips)
    return draw_random_subclip(noise_clip, n_samples)


def preprocess_wav(wav, distort=True):
    if distort:
        wav = randomly_distort_wavfile(wav, sr=16000)
    wav = fix_wavfile_length(wav, 16000)
    wav = normalize_wavfile(wav, normalize_to_peak=True)
    return wav.astype(np.float32)


def generate_augmented_wav(filepath, folder_name="augmented", suffix=""):
    try:
        folder_class = os.path.split(os.path.split(filepath)[-2])[-1]
        output_path = os.path.join(get_training_data_path(), folder_name, folder_class)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        sample_rate, wav = read_wavfile(filepath)
        wav = preprocess_wav(wav)
        # Add suffix
        filename = os.path.split(filepath)[-1]
        filename = os.path.splitext(filename)[0] + "_" + suffix + os.path.splitext(filename)[1]
        output_filepath = os.path.join(output_path, filename)
        wavfile.write(output_filepath, sample_rate, wav)
    except:
        print("Error found with file {}".format(filepath))


def batch_augment_files(list_of_files, n_times, n_jobs, folder_name="augmented"):
    for i in range(n_times):
        list_of_files = list_of_files[:]
        random.shuffle(list_of_files)
        if n_jobs > 1:
            pool = Pool(n_jobs)
            pool.map(lambda x: generate_augmented_wav(filepath=x,
                                                      folder_name=folder_name,
                                                      suffix=str(i)), list_of_files)
            pool.close()
            pool.join()
        else:
            list(map(lambda x: generate_augmented_wav(filepath=x,
                                                      folder_name=folder_name,
                                                      suffix=str(i)), list_of_files))
