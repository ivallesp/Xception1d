import os
import random
import tarfile
import urllib.request
from multiprocessing.dummy import Pool

import numpy as np
import torch
from scipy.io import wavfile
from tqdm import tqdm

from src.common_paths import get_training_data_path, get_dataset_filepath, get_augmented_data_path
from src.data_tools import read_wavfile, draw_random_subclip, randomly_distort_wavfile, fix_wavfile_length, \
    normalize_wavfile
from src.general_utilities import batching, flatten, recursive_listdir


def download_dataset():
    """
    Download the data and stores the tar.gz file in the specified path
    """
    url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    urllib.request.urlretrieve(url, get_dataset_filepath())


def decompress_dataset():
    """
    Retrieves the downloaded data and decompresses it
    """
    fname = get_dataset_filepath()
    assert os.path.exists(fname)
    tar = tarfile.open(fname, "r:gz")
    tar.extractall(path=get_training_data_path())
    tar.close()


def get_list_of_wav_paths(include_augmentations: bool = False) -> tuple:
    """
    Retrieves the list of filepaths that belong to train, validation and test
    :param include_augmentations: if set, the code will look for files in the augmented folder. These files will be add
    to the training list (bool)
    :return: list of training paths, list of validation paths and list of test paths (list of lists)
    """
    folders = [get_training_data_path()]
    folders += [get_augmented_data_path()] if include_augmentations else []

    list_test = open(os.path.join(get_training_data_path(), "testing_list.txt"))
    list_test = list(map(lambda x: os.path.normpath(os.path.join(get_training_data_path(), x.strip())), list_test))

    list_val = open(os.path.join(get_training_data_path(), "validation_list.txt"))
    list_val = list(map(lambda x: os.path.normpath(os.path.join(get_training_data_path(), x.strip())), list_val))

    list_train = flatten([list(recursive_listdir(os.path.normpath(folder))) for folder in folders])
    list_train = list(filter(lambda p: "background_noise" not in p and p.endswith("wav"), list_train))
    list_train = np.setdiff1d(list_train, list_test + list_val).tolist()
    return list_train, list_val, list_test


def generate_white_noise_clip(n_samples: int) -> np.array:
    """
    Generates an artificial silence clip using white noise and returns it
    :param n_samples: number of samples of the desired clip (int)
    :return: clip (np.array)
    """
    clip = np.random.randn(n_samples)
    clip /= np.abs(clip).max()
    return clip


def load_real_noise_clips() -> np.array:
    """
    Loads all the available real noise clips, which are located under the _background_noise_ folder
    :return: list of real noise clips (list of np.array)
    """
    clips = []
    path = os.path.join(get_training_data_path(), "_background_noise_")
    for filename in filter(lambda x: x.endswith(".wav"), os.listdir(path)):
        _, wav = read_wavfile(os.path.join(path, filename))
        clips.append(wav)
    return clips


def load_random_real_noise_clip() -> np.array:
    """
    Loads a random noise clip from the _background_noise_ folder and returns it
    :return: real noise clip (np.array)
    """
    path = os.path.join(get_training_data_path(), "_background_noise_")
    filename = random.choice(list(filter(lambda x: x.endswith(".wav"), os.listdir(path))))
    _, clip = read_wavfile(os.path.join(path, filename))
    return clip


def get_random_real_noise_subclip(n_samples: int, noise_clips: list = None) -> np.array:
    """
    Loads a random noise clip and cuts it to the desired size
    :param n_samples: number of samples in the desired clip (int)
    :param noise_clips: list of pre-loaded noise clips. If passed, the function selects one randomly instead of loading
    it from scratch (list of np.array)
    :return: a clip of the desired size (np.array)
    """
    if noise_clips is None:
        noise_clip = load_random_real_noise_clip()
    else:
        noise_clip = random.choice(noise_clips)
    return draw_random_subclip(noise_clip, n_samples)


def preprocess_wav(wav: np.array, distort: bool = True) -> np.array:
    """
    Gets a wav clip and (optionally) distorts it randomly, fixes the length of the clip, normalizes it and casts it
    to type np.float32
    :param wav: audio clip (np.array)
    :param distort: indicates if the clip must be distorted (bool)
    :return: the preprocessed clip (np.array)
    """
    if distort:
        wav = randomly_distort_wavfile(wav, sr=16000)
    wav = fix_wavfile_length(wav, 16000)
    wav = normalize_wavfile(wav, normalize_to_peak=True)
    return wav.astype(np.float32)


def generate_augmented_wav(filepath: str, suffix: str = "") -> None:
    """
    Given a filepath of a wav file, loads it, preprocesses it and stores it in the default folder for augmentations.
    :param filepath:  file path of an existing wav file (str)
    :param suffix: piece of text to be appended before the extension in the output filepath. It is not needed to add
     a separator at the begining, "_" is added by default(str)
    :return: None (void)
    """
    try:
        folder_class = os.path.split(os.path.split(filepath)[-2])[-1]
        output_path = os.path.join(get_augmented_data_path(), folder_class)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        sample_rate, wav = read_wavfile(filepath)
        wav = preprocess_wav(wav, distort=True)
        # Add suffix
        filename = os.path.split(filepath)[-1]
        filename = os.path.splitext(filename)[0] + "_" + suffix + os.path.splitext(filename)[1]
        output_filepath = os.path.join(output_path, filename)
        wavfile.write(output_filepath, sample_rate, wav)
    except:
        print("Error found with file {}".format(filepath))


def batch_augment_files(list_of_files: list, n_times: int, n_jobs: int = 1) -> None:
    for i in range(n_times):
        list_of_files = list_of_files[:]
        random.shuffle(list_of_files)
        if n_jobs > 1:
            pool = Pool(n_jobs)
            pool.map(lambda x: generate_augmented_wav(filepath=x,
                                                      suffix=str(i)), list_of_files)
            pool.close()
            pool.join()
        else:
            list(map(lambda x: generate_augmented_wav(filepath=x,
                                                      suffix=str(i)), list_of_files))


class DataFeeder:
    def __init__(self, file_paths: str, batch_size: int, add_noise: bool = False, shuffle: bool = True,
                 scoring: bool = False) -> None:
        self.known_classes = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown",
                              "silence"]
        self.scoring = scoring
        self.shuffle = shuffle
        if not scoring:
            self.target_encoder = dict(zip(self.known_classes, range(len(self.known_classes))))
            self.target_decoder = {v: k for k, v in self.target_encoder.items()}
        self.file_paths = file_paths
        self.corrupted_file_paths = []
        self.set_batch_size(batch_size)
        self.noise_clips = load_real_noise_clips()
        self.load_data(file_paths, add_noise=add_noise, load_targets=not scoring)
        if shuffle:
            self.shuffle_data()
        self.prepare_data()
        assert not np.isnan(self.audios).any()
        assert self.audios.max() <= 1
        assert self.audios.min() >= -1

    def prepare_data(self) -> None:
        self.audios = np.array(self.audios)
        if not self.scoring:
            self.targets = [target if target in self.known_classes else "unknown" for target in self.targets]
            self.targets = list(map(self.target_encoder.get, self.targets))
            self.targets = np.array(self.targets)

    def shuffle_data(self) -> None:
        joined_list = list(zip(self.audios, self.targets))
        random.shuffle(joined_list)
        self.audios, self.targets = list(zip(*joined_list))  # Shuffle!

    def load_data(self, file_paths: list, add_noise: bool, load_targets: bool = True) -> None:
        self.audios = []
        if load_targets:
            self.targets = []
        # Load data
        for file_path in tqdm(file_paths):
            if os.path.exists(file_path):
                try:
                    _, wav = read_wavfile(file_path)
                    wav = preprocess_wav(wav, distort=False)
                    self.audios.append(wav)
                    if load_targets:
                        target = os.path.split(os.path.split(file_path)[0])[1]
                        self.targets.append(target)
                except:
                    self.corrupted_file_paths.append(file_path)
                    print(f"Error reading {file_path}")
            else:
                print(f"Fatal error, file {file_path} not found")
        if add_noise:
            n_artificial_noise_samples = int(0.05 * len(self.audios))
            n_real_noise_samples = int(0.15 * len(self.audios))
            n_empty_samples = int(0.0025 * len(self.audios))
            for i in tqdm(range(n_real_noise_samples)):
                wav = get_random_real_noise_subclip(n_samples=16000, noise_clips=self.noise_clips)
                wav = preprocess_wav(wav, distort=False)
                self.audios.append(wav)
                if load_targets: self.targets.append("silence")
            for i in range(n_artificial_noise_samples):
                wav = generate_white_noise_clip(16000)
                wav = preprocess_wav(wav, distort=False)
                self.audios.append(wav)
                if load_targets: self.targets.append("silence")
            for i in range(n_empty_samples):
                self.audios.append(np.zeros_like(wav).astype(np.float32))
                if load_targets: self.targets.append("silence")

    def get_batches(self, return_incomplete_batches: bool = False):
        list_of_iterables = [self.audios, self.targets] if not self.scoring else [self.audios]
        for batch in batching(list_of_iterables=list_of_iterables,
                              n=self.batch_size,
                              return_incomplete_batches=return_incomplete_batches):
            batch[0] = np.expand_dims(np.array(batch[0]), 1)
            batch[0] = torch.from_numpy(batch[0])
            if self.scoring:
                batch += [None]
            else:
                batch[1] = torch.from_numpy(batch[1])
            yield batch

    def set_batch_size(self, batch_size) -> None:
        self.batch_size = batch_size
