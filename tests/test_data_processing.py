import os
import shutil
from unittest import TestCase

import numpy as np

from src.common_paths import get_dataset_filepath, get_augmented_data_path, get_augmented_data_folder
from src.data_processing import get_list_of_wav_paths, generate_white_noise_clip, load_real_noise_clips, \
    get_random_real_noise_subclip, load_random_real_noise_clip, preprocess_wav, generate_augmented_wav, \
    batch_augment_files, DataFeeder, download_dataset, decompress_dataset
from src.data_tools import read_wavfile


class TestDataProcessing(TestCase):
    def setUp(self):
        self.wav_filepath = "./tests/examples/testaudio.wav"
        self.data_version = "0.02"
        if not os.path.exists(get_dataset_filepath(data_version=self.data_version)):
            download_dataset(data_version=self.data_version)
            decompress_dataset(data_version=self.data_version)
        self.known_commands = ["zero", "one", "two", "three", "four", "five", "six",
                               "seven", "eight", "nine", "marvin", "sheila", "forward",
                               "backward", "bed", "bird", "cat", "dog", "down", "follow",
                               "go", "happy", "house", "learn", "no", "yes", "off",
                               "on", "right", "left", "stop", "tree", "up", "visual", "wow"]

    def tearDown(self):
        pass

    def test_get_list_of_wav_paths(self):
        train, val, test = get_list_of_wav_paths(data_version=self.data_version)
        whole_list = train + val + test
        self.assertEqual(len(whole_list), len(set(whole_list)))
        self.assertEqual(len(whole_list), 105829)
        self.assertEqual(9981, len(val))
        self.assertEqual(11005, len(test))
        self.assertTrue(all([os.path.exists(fn) for fn in whole_list]))
        set_train = set([x.split(os.sep)[-2] for x in train])
        set_val = set([x.split(os.sep)[-2] for x in val])
        set_test = set([x.split(os.sep)[-2] for x in test])
        self.assertTrue(set_train == set_val)
        self.assertTrue(set_train == set_test)

    def test_get_list_of_wav_paths_augmented(self):
        train, val, test = get_list_of_wav_paths(data_version=self.data_version)
        train_aug, val_aug, test_aug = get_list_of_wav_paths(data_version=self.data_version, n_augmentations="all")
        self.assertGreaterEqual(len(train_aug), len(train))
        self.assertEqual(len(val_aug), len(val))
        self.assertEqual(len(test_aug), len(test))

    def test_generate_white_noise_clip(self):
        clip = generate_white_noise_clip(300)
        self.assertGreater(4, np.abs(clip).max())
        self.assertLess(1, len(np.unique(clip)))
        self.assertEqual(300, len(clip))

    def test_load_real_noise_clips(self):
        noise_clips = load_real_noise_clips(data_version=self.data_version)
        self.assertEqual(6, len(noise_clips))
        for noise_clip in noise_clips:
            self.assertLess(1000, len(noise_clip))

    def test_get_random_real_noise_subclip(self):
        prev_noise_clip = get_random_real_noise_subclip(data_version=self.data_version, n_samples=333)
        for _ in range(10):
            noise_clip = get_random_real_noise_subclip(data_version=self.data_version, n_samples=333)
            self.assertEqual(333, len(noise_clip))
            self.assertLess(0.5, np.max(np.array(noise_clip) != 0))
            self.assertNotEqual(np.abs(prev_noise_clip).sum(), np.abs(noise_clip).sum())
            prev_noise_clip = noise_clip
        noise_clips = load_real_noise_clips(data_version=self.data_version)
        prev_noise_clip = get_random_real_noise_subclip(data_version=self.data_version, n_samples=333,
                                                        noise_clips=noise_clips)
        for _ in range(10):
            noise_clip = get_random_real_noise_subclip(data_version=self.data_version, n_samples=333,
                                                       noise_clips=noise_clips)
            self.assertEqual(333, len(noise_clip))
            self.assertLess(0.5, np.max(np.array(noise_clip) != 0))
            self.assertNotEqual(np.abs(prev_noise_clip).sum(), np.abs(noise_clip).sum())
            prev_noise_clip = noise_clip

    def test_load_random_real_noise_clip(self):
        clip = load_random_real_noise_clip(data_version=self.data_version)
        self.assertLess(1000, len(clip))
        self.assertLess(0.5, np.max(np.array(clip) != 0))

    def test_preprocess_wav(self):
        _, wav = read_wavfile(self.wav_filepath)
        wav_distorted = preprocess_wav(wav, distort=True)
        wav_preprocessed = preprocess_wav(wav, distort=False)
        self.assertEqual(16000, len(wav_distorted))
        self.assertEqual(16000, len(wav_preprocessed))
        self.assertGreaterEqual(1, np.abs(wav_distorted).max())
        self.assertGreaterEqual(1, np.abs(wav_preprocessed).max())
        self.assertLess(500, len(set(wav_preprocessed)))
        self.assertLess(500, len(set(wav_distorted)))

    def test_generate_augmented_wav(self):
        filepath_augmented = os.path.join(get_augmented_data_path(data_version=self.data_version), "testfolder", "examples",
                                          "testaudio.wav")
        if os.path.exists(filepath_augmented):
            os.remove(filepath_augmented)
        generate_augmented_wav(data_version=self.data_version, filepath=self.wav_filepath, folder="testfolder")
        self.assertTrue(os.path.exists(filepath_augmented))
        _, wav = read_wavfile(filepath_augmented)
        self.assertEqual(16000, len(wav))
        self.assertGreaterEqual(1, np.abs(wav).max())
        os.remove(filepath_augmented)
        shutil.rmtree(os.path.split(os.path.split(filepath_augmented)[0])[0])

    def test_batch_augment_files(self):
        list_of_files = [self.wav_filepath]
        for i in range(10):
            for filepath in list_of_files:
                folder_name = "test_"+str(i)
                batch_augment_files(data_version=self.data_version, list_of_files=list_of_files,
                                    folder_name=folder_name, n_jobs=1)
                path, filename = os.path.split(filepath)
                _, folder_class = os.path.split(path)
                output_filepath = os.path.join(get_augmented_data_folder(data_version=self.data_version,
                                                                         folder=folder_name), folder_class, filename)
                self.assertTrue(os.path.exists(output_filepath))
                shutil.rmtree(os.path.split(os.path.split(output_filepath)[0])[0])

    def test_batch_augment_files_multiprocessing(self):
        list_of_files = [self.wav_filepath]
        for i in range(10):
            for filepath in list_of_files:
                folder_name = "test_"+str(i)
                batch_augment_files(data_version=self.data_version, list_of_files=list_of_files,
                                    folder_name=folder_name, n_jobs=2)
                path, filename = os.path.split(filepath)
                _, folder_class = os.path.split(path)
                output_filepath = os.path.join(get_augmented_data_folder(data_version=self.data_version,
                                                                         folder=folder_name), folder_class, filename)
                self.assertTrue(os.path.exists(output_filepath))
                shutil.rmtree(os.path.split(os.path.split(output_filepath)[0])[0])

    def test_data_feeder(self):
        train, _, _ = get_list_of_wav_paths(data_version=self.data_version)
        list_of_files = train[0:20] + ["/test/errors.wav"]
        df = DataFeeder(data_version=self.data_version, file_paths=list_of_files, batch_size=5, include_silence=False,
                        known_commands=self.known_commands, include_unknown=True)
        self.assertEqual(20, len(df.audios))
        self.assertEqual(20, len(df.targets))
        list_of_batches = list(df.get_batches())
        self.assertEqual(4, len(list_of_batches))
        self.assertEqual(2, len(list_of_batches[0]))
        self.assertLessEqual(1, np.max(np.abs(df.audios)))
        self.assertIsNotNone(list_of_batches[0][1])
        self.assertGreaterEqual(35, np.max(df.targets))
        self.assertIn("/test/errors.wav", df.corrupted_file_paths)

    def test_data_feeder_with_silence(self):
        train, _, _ = get_list_of_wav_paths(data_version=self.data_version)
        list_of_files = train[0:60] + ["/test/errors.wav"]
        df = DataFeeder(data_version=self.data_version, file_paths=list_of_files, batch_size=5, include_silence=True,
                        known_commands=self.known_commands + ["test"], include_unknown=False)
        self.assertEqual(60 + 12, len(df.audios))
        self.assertEqual(60 + 12, len(df.targets))
        list_of_batches = list(df.get_batches())
        self.assertEqual(14, len(list_of_batches))
        self.assertEqual(2, len(list_of_batches[0]))
        self.assertLessEqual(1, np.max(np.abs(df.audios)))
        self.assertIsNotNone(list_of_batches[0][1])
        self.assertGreaterEqual(36, np.max(df.targets))
        self.assertIn("/test/errors.wav", df.corrupted_file_paths)

    def test_data_feeder_real_with_unk(self):
        _, validation, _ = get_list_of_wav_paths(data_version=self.data_version)
        df = DataFeeder(data_version=self.data_version,
                        file_paths=validation,
                        batch_size=5,
                        include_silence=False,
                        known_commands=["left", "right"],
                        include_unknown=True)
        self.assertEqual(len(df.targets), len(df.audios))
        self.assertEqual(9981, len(df.audios))
        self.assertEqual({0, 1, 2}, set(df.targets.tolist()))

    def test_data_feeder_real_with_unk_and_silence(self):
        _, validation, _ = get_list_of_wav_paths(data_version=self.data_version)
        df = DataFeeder(data_version=self.data_version,
                        file_paths=validation,
                        batch_size=5,
                        include_silence=True,
                        known_commands=["left", "right"],
                        include_unknown=True)
        self.assertEqual(len(df.targets), len(df.audios))
        self.assertEqual(9981 + int(9981 * 0.15) + int(9981 * 0.05) + int(9981 * 0.0025), len(df.audios))
        self.assertEqual({0, 1, 2, 3}, set(df.targets.tolist()))

    def test_data_feeder_real_with_no_unk(self):
        _, validation, _ = get_list_of_wav_paths(data_version=self.data_version)
        df = DataFeeder(data_version=self.data_version,
                        file_paths=validation,
                        batch_size=5,
                        include_silence=False,
                        known_commands=["left", "right"],
                        include_unknown=False)
        self.assertEqual(len(df.targets), len(df.audios))
        self.assertEqual(715, len(df.audios))
        self.assertEqual({0, 1}, set(df.targets.tolist()))
