import os
import random
from multiprocessing.dummy import Pool

import numpy as np
import torch
from scipy.io import wavfile
from tqdm import tqdm

from src.common_paths import get_training_data_path
from src.data_tools import read_wavfile, draw_random_subclip, randomly_distort_wavfile, fix_wavfile_length, \
    normalize_wavfile
from src.general_utilities import batching


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


class DataFeeder:
    def __init__(self, file_paths, batch_size, add_noise=False):
        self.known_classes = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown",
                              "silence"]
        self.target_encoder = dict(zip(self.known_classes, range(len(self.known_classes))))
        self.target_decoder = {v: k for k, v in self.target_encoder.items()}
        self.file_paths = file_paths
        self.set_batch_size(batch_size)
        self.noise_clips = load_real_noise_clips()
        self.load_data(file_paths, add_noise=add_noise)
        self.shuffle_data()
        self.prepare_data()
        assert not np.isnan(self.audios).any()
        assert self.audios.max() <= 1
        assert self.audios.min() >= -1
        
    def prepare_data(self):
        self.targets = [target if target in self.known_classes else "unknown" for target in self.targets]
        self.targets = list(map(self.target_encoder.get, self.targets))
        self.audios = np.array(self.audios)
        self.targets = np.array(self.targets)

    def shuffle_data(self):
        joined_list = list(zip(self.audios, self.targets))
        random.shuffle(joined_list)
        self.audios, self.targets = list(zip(*joined_list))  # Shuffle!

    def load_data(self, file_paths, add_noise):
        self.audios = []
        self.targets = []
        # Load data
        for file_path in tqdm(file_paths):
            if os.path.exists(file_path):
                try:
                    _, wav = read_wavfile(file_path)
                    wav = preprocess_wav(wav, distort=False)
                    target = os.path.split(os.path.split(file_path)[0])[1]
                    self.audios.append(wav)
                    self.targets.append(target)
                except:
                    print(f"Error reading {file_path}")
            else:
                print(f"Fatal error, file {file_path} not found")
        if add_noise:
            n_artificial_noise_samples = int(0.05 * len(self.audios))
            n_real_noise_samples = int(0.15 * len(self.audios))
            for i in tqdm(range(n_real_noise_samples)):
                wav = get_random_real_noise_subclip(n_samples=16000, noise_clips=self.noise_clips)
                wav = preprocess_wav(wav, distort=False)
                self.audios.append(wav)
                self.targets.append("silence")
            for i in range(n_artificial_noise_samples):
                wav = generate_white_noise_clip(16000)
                wav = preprocess_wav(wav, distort=False)
                self.audios.append(wav)
                self.targets.append("silence")

    def get_batches(self, return_incomplete_batches=False):
        for batch in batching(list_of_iterables=[self.audios, self.targets],
                              n=self.batch_size,
                              return_incomplete_batches=return_incomplete_batches):
            batch[0] = np.expand_dims(np.array(batch[0]), 1)
            batch[0] = torch.from_numpy(batch[0])
            batch[1] = torch.from_numpy(batch[1])
            yield batch

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
