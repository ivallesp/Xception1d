import os
import shutil
import unittest
from unittest import TestCase

import numpy as np

from src.common_paths import get_training_data_path
from src.data_processing import get_list_of_wav_paths, generate_white_noise_clip, load_real_noise_clips, \
    get_random_real_noise_subclip, load_random_real_noise_clip, preprocess_wav, generate_augmented_wav, \
    batch_augment_files, DataFeeder
from src.data_tools import read_wavfile


class TestDataProcessing(TestCase):
    def setUp(self):
        self.wav_filepath = "./tests/examples/testaudio.wav"

    def tearDown(self):
        pass

    @unittest.skipIf("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", "Skipping this test on Travis CI.")
    def test_get_list_of_wav_paths(self):
        train, val, test, scoring = get_list_of_wav_paths()
        whole_list = train + val + test
        self.assertEqual(len(whole_list), len(set(whole_list)))
        self.assertGreater(len(train), 50000)
        self.assertGreater(len(val), 5000)
        self.assertGreater(len(test), 5000)
        self.assertEqual(64727 - 6, len(whole_list))
        self.assertEqual(158538, len(scoring))
        self.assertTrue(all([os.path.exists(fn) for fn in whole_list]))
        self.assertTrue(all([os.path.exists(fn) for fn in scoring]))
        set_train = set([x.split(os.sep)[-2] for x in train])
        set_val = set([x.split(os.sep)[-2] for x in val])
        set_test = set([x.split(os.sep)[-2] for x in test])
        self.assertTrue(set_train == set_val)
        self.assertTrue(set_train == set_test)
        train_aug, val_aug, test_aug, scoring_aug = get_list_of_wav_paths(include_augmentations=True)
        self.assertGreaterEqual(len(train_aug), len(train))
        self.assertEqual(len(val_aug), len(val))
        self.assertEqual(len(test_aug), len(test))
        self.assertEqual(len(scoring_aug), len(scoring))


    def test_generate_white_noise_clip(self):
        clip = generate_white_noise_clip(300)
        self.assertGreater(4, np.abs(clip).max())
        self.assertLess(1, len(np.unique(clip)))
        self.assertEqual(300, len(clip))

    @unittest.skipIf("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", "Skipping this test on Travis CI.")
    def test_load_real_noise_clips(self):
        noise_clips = load_real_noise_clips()
        self.assertEqual(6, len(noise_clips))
        for noise_clip in noise_clips:
            self.assertLess(1000, len(noise_clip))

    @unittest.skipIf("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", "Skipping this test on Travis CI.")
    def test_get_random_real_noise_subclip(self):
        prev_noise_clip = get_random_real_noise_subclip(333)
        for _ in range(10):
            noise_clip = get_random_real_noise_subclip(333)
            self.assertEqual(333, len(noise_clip))
            self.assertLess(0.5, np.max(np.array(noise_clip) != 0))
            self.assertNotEqual(np.abs(prev_noise_clip).sum(), np.abs(noise_clip).sum())
            prev_noise_clip = noise_clip
        noise_clips = load_real_noise_clips()
        prev_noise_clip = get_random_real_noise_subclip(333, noise_clips)
        for _ in range(10):
            noise_clip = get_random_real_noise_subclip(333, noise_clips)
            self.assertEqual(333, len(noise_clip))
            self.assertLess(0.5, np.max(np.array(noise_clip) != 0))
            self.assertNotEqual(np.abs(prev_noise_clip).sum(), np.abs(noise_clip).sum())
            prev_noise_clip = noise_clip

    @unittest.skipIf("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", "Skipping this test on Travis CI.")
    def test_load_random_real_noise_clip(self):
        clip = load_random_real_noise_clip()
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
        filepath_augmented = os.path.join(get_training_data_path(), "test_augmentation",
                                          "examples", "testaudio_testsuffix.wav")
        if os.path.exists(filepath_augmented):
            os.remove(filepath_augmented)
        generate_augmented_wav(self.wav_filepath, "test_augmentation", suffix="testsuffix")
        self.assertTrue(os.path.exists(filepath_augmented))
        _, wav = read_wavfile(filepath_augmented)
        self.assertEqual(16000, len(wav))
        self.assertGreaterEqual(1, np.abs(wav).max())
        os.remove(filepath_augmented)
        os.rmdir(os.path.split(filepath_augmented)[0])

    @unittest.skipIf("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", "Skipping this test on Travis CI.")
    def test_batch_augment_files(self):
        list_of_files, _, _, _ = get_list_of_wav_paths()
        list_of_files = list_of_files[:5]
        batch_augment_files(list_of_files=list_of_files, n_times=10, n_jobs=1, folder_name="test_augmentation")
        for i in range(10):
            for filepath in list_of_files:
                path, filename = os.path.split(filepath)
                _, folder = os.path.split(path)
                name, extension = os.path.splitext(filename)
                output_filename = name + "_" + str(i) + extension
                output_filepath = os.path.join(get_training_data_path(), "test_augmentation", folder, output_filename)
                self.assertTrue(os.path.exists(output_filepath))
                os.remove(output_filepath)

        shutil.rmtree(os.path.join(get_training_data_path(), "test_augmentation"))

    @unittest.skipIf("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", "Skipping this test on Travis CI.")
    def test_data_feeder(self):
        train, _, _, _ = get_list_of_wav_paths()
        list_of_files = train[0:20] + ["/test/errors"]
        df = DataFeeder(file_paths=list_of_files, batch_size=5)
        self.assertEqual(20, len(df.audios))
        self.assertEqual(20, len(df.targets))
        list_of_batches = list(df.get_batches())
        self.assertEqual(4, len(list_of_batches))
        self.assertEqual(2, len(list_of_batches[0]))
        self.assertLessEqual(1, np.max(np.abs(df.audios)))
        self.assertIsNotNone(list_of_batches[0][1])
        self.assertGreaterEqual(11, np.max(df.targets))

    @unittest.skipIf("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", "Skipping this test on Travis CI.")
    def test_data_feeder_with_scoring(self):
        _, _, _, scoring = get_list_of_wav_paths()
        list_of_files = scoring[0:1000] + ["/test/errors"]
        df = DataFeeder(file_paths=list_of_files, batch_size=5, add_noise=False, shuffle=False, scoring=True)
        self.assertEqual(1000, len(df.audios))
        self.assertNotIn("targets", dir(df))
        list_of_batches = list(df.get_batches())
        self.assertEqual(200, len(list_of_batches))
        self.assertEqual(2, len(list_of_batches[0]))
        self.assertIsNone(list_of_batches[0][1])
        self.assertGreaterEqual(1, np.max(np.abs(df.audios)))
        self.assertLess(0, np.max(np.abs(df.audios)))
