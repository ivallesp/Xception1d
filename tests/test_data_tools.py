from unittest import TestCase

import numpy as np

from src.data_tools import read_wavfile, normalize_wavfile, fix_wavfile_duration, \
    fix_wavfile_length, saturate_wavfile, resample_wavfile, time_offset_wavfile, add_noise_to_wavfile, \
    pitch_shift_wavfile, randomly_distort_wavfile, draw_random_subclip


class TestDataTools(TestCase):
    def setUp(self):
        self.wav_filepath = "./tests/examples/testaudio.wav"

    def tearDown(self):
        pass

    def test_read_wavfile(self):
        sample_rate, wav = read_wavfile(self.wav_filepath)
        self.assertEqual(16000, sample_rate)
        self.assertEqual(len(wav), 14118)
        # 16-bit PCM
        self.assertGreaterEqual(min(wav), -32768)
        self.assertLessEqual(max(wav), 32768)

    def test_normalize_wavfile_formats(self):
        sample_rate, wav = read_wavfile(self.wav_filepath)
        wav_0 = wav / 32768
        wav_1 = wav_0 * 2147483648
        wav_2 = wav
        wav_3 = 255 * (wav_0 + 1) / 2
        wav_0_processed = normalize_wavfile(wav_0, 0)
        wav_1_processed = normalize_wavfile(wav_1, 1)
        wav_2_processed = normalize_wavfile(wav_2, 2)
        wav_3_processed = normalize_wavfile(wav_3, 3)
        self.assertTrue(max(wav_0_processed) == max(wav_1_processed) == max(wav_2_processed) == max(wav_3_processed))
        self.assertTrue(min(wav_0_processed) == min(wav_1_processed) == min(wav_2_processed) == min(wav_3_processed))
        self.assertLessEqual(max(wav_0_processed), 1)
        self.assertAlmostEqual(max(wav_0_processed), 1, delta=0.75)
        self.assertGreaterEqual(min(wav_0_processed), -1)
        self.assertAlmostEqual(min(wav_0_processed), -1, delta=0.75)
        with self.assertRaises(ValueError):
            normalize_wavfile(np.zeros(16000), normalize_to_peak=True) 
        with self.assertRaises(ValueError):
            wav_corrupted = np.random.rand(16000)
            wav_corrupted[4000] = np.nan
            normalize_wavfile(wav_corrupted, normalize_to_peak=True)
        with self.assertRaises(ValueError):
            normalize_wavfile(np.ones(16000)*np.nan, normalize_to_peak=True) 
    
    def test_normalize_wavfile_to_peaks(self):
        sample_rate, wav = read_wavfile(self.wav_filepath)
        wav_processed = normalize_wavfile(wav, normalize_to_peak=True)
        self.assertGreaterEqual(min(wav_processed), -1)
        self.assertLessEqual(max(wav_processed), 1)
        self.assertEqual(max(max(wav_processed), -min(wav_processed)), 1)

    def test_fix_wavfile_duration(self):
        sample_rate, wav = read_wavfile(self.wav_filepath)

        wav_processed = fix_wavfile_duration(wav=wav, sample_rate=sample_rate, duration=len(wav) / sample_rate)
        self.assertListEqual(list(wav), list(wav_processed))

        wav_processed = fix_wavfile_duration(wav=wav, sample_rate=sample_rate, duration=0.3)
        self.assertEqual(len(wav_processed), int(0.3 * sample_rate))

        wav_processed = fix_wavfile_duration(wav=wav, sample_rate=sample_rate, duration=3)
        self.assertEqual(len(wav_processed), int(3 * sample_rate))
        self.assertListEqual([0] * 16000, list(wav_processed[:16000]))
        self.assertListEqual([0] * 16000, list(wav_processed[-16000:]))
        self.assertEqual(np.sum(wav), np.sum(wav_processed))

    def test_fix_wavfile_length(self):
        sample_rate, wav = read_wavfile(self.wav_filepath)
        wav_processed = fix_wavfile_length(wav, 8000)  # Cut
        self.assertEqual(8000, len(wav_processed))
        self.assertAlmostEqual(wav_processed.mean(), wav.mean(), delta=0.05)
        wav_processed = fix_wavfile_length(wav, 30000)  # Pad
        self.assertEqual(30000, len(wav_processed))
        self.assertEqual(wav_processed.sum(), wav.sum())
        self.assertListEqual([0] * 1000, wav_processed[:1000].tolist())
        self.assertListEqual([0] * 1000, wav_processed[-1000:].tolist())
        wav_processed = fix_wavfile_length(wav, 30001)  # Pad odd
        self.assertEqual(30001, len(wav_processed))
        self.assertEqual(wav_processed.sum(), wav.sum())
        self.assertListEqual([0] * 1000, wav_processed[:1000].tolist())
        self.assertListEqual([0] * 1000, wav_processed[-1000:].tolist())

    def test_saturate_wavfile(self):
        sample_rate, wav = read_wavfile(self.wav_filepath)
        wav = normalize_wavfile(wav, normalize_to_peak=True)
        wav_processed = saturate_wavfile(wav, 1)
        self.assertListEqual(wav.tolist(), wav_processed.tolist())
        wav_processed = saturate_wavfile(wav, 2)  # Atenuate
        self.assertAlmostEqual(wav.mean(), wav_processed.mean(), delta=0.05)
        self.assertGreater(np.std(wav), np.std(wav_processed))
        self.assertGreater(max(abs(wav)), max(abs(wav_processed)))
        wav_processed = saturate_wavfile(wav, 0.2)  # Saturate
        self.assertAlmostEqual(wav.mean(), wav_processed.mean(), delta=0.05)
        self.assertLess(np.std(wav), np.std(wav_processed))
        self.assertEqual(max(abs(wav)), max(abs(wav_processed)))

    def test_resample_wavfile(self):
        sample_rate, wav = read_wavfile(self.wav_filepath)
        wav_processed = resample_wavfile(wav=wav, factor=1)
        self.assertEqual(len(wav), len(wav_processed))
        self.assertAlmostEqual(wav.sum(), wav_processed.sum(), delta=0.00001)
        wav_processed = resample_wavfile(wav=wav, factor=2)
        self.assertEqual(len(wav), len(wav_processed))
        self.assertLess(np.std(wav), np.std(wav_processed))
        wav_processed = resample_wavfile(wav=wav, factor=0.5)
        self.assertEqual(len(wav), len(wav_processed))
        self.assertGreater(np.std(wav), np.std(wav_processed))

    def test_time_offset(self):
        sample_rate, wav = read_wavfile(self.wav_filepath)
        wav_processed = time_offset_wavfile(wav=wav, shift_factor=0)
        self.assertEqual(len(wav), len(wav_processed))
        self.assertAlmostEqual(wav.sum(), wav_processed.sum(), delta=0.00001)
        wav_processed = time_offset_wavfile(wav=wav, shift_factor=1)  # Full_shift
        self.assertEqual(np.abs(wav_processed).sum(), 0)
        wav_processed = time_offset_wavfile(wav=wav, shift_factor=-1)  # Full shift inverse
        self.assertEqual(np.abs(wav_processed).sum(), 0)
        wav_processed = time_offset_wavfile(wav=wav, shift_factor=0.5)  # Half shift
        self.assertEqual(len(wav), len(wav_processed))
        self.assertListEqual(wav.tolist()[:int(len(wav) / 2)], wav_processed.tolist()[-int(len(wav) / 2):])
        wav_processed = time_offset_wavfile(wav=wav, shift_factor=-0.5)  # Half shift inverse
        self.assertEqual(len(wav), len(wav_processed))
        self.assertListEqual(wav.tolist()[-int(len(wav) / 2):], wav_processed.tolist()[:int(len(wav) / 2)])

    def test_add_noise(self):
        sample_rate, wav = read_wavfile(self.wav_filepath)
        wav_processed = add_noise_to_wavfile(wav=wav, amplitude_factor=0, clip_to_original_range=False)
        self.assertEqual(len(wav), len(wav_processed))
        self.assertAlmostEqual(wav.sum(), wav_processed.sum(), delta=0.00001)
        wav_processed = add_noise_to_wavfile(wav=wav, amplitude_factor=0, clip_to_original_range=True)
        self.assertEqual(len(wav), len(wav_processed))
        self.assertAlmostEqual(wav.sum(), wav_processed.sum(), delta=0.00001)

        wav_processed = add_noise_to_wavfile(wav=wav, amplitude_factor=0.5, clip_to_original_range=False)
        self.assertEqual(len(wav), len(wav_processed))
        self.assertNotEquals(np.abs(wav).max(), np.abs(wav_processed).max())
        self.assertLess(np.std(wav), np.std(wav_processed))
        wav_processed = add_noise_to_wavfile(wav=wav, amplitude_factor=0.5, clip_to_original_range=True)
        self.assertEqual(len(wav), len(wav_processed))
        self.assertEquals(np.abs(wav).max(), np.abs(wav_processed).max())

    def test_pitch_shift(self):
        sample_rate, wav = read_wavfile(self.wav_filepath)
        wav_processed = pitch_shift_wavfile(wav=wav, sr=sample_rate, n_octaves=0)
        self.assertEqual(len(wav), len(wav_processed))
        self.assertAlmostEqual(wav.sum(), wav_processed.sum(), delta=0.00001)

        wav_processed = pitch_shift_wavfile(wav=wav, sr=sample_rate, n_octaves=0.5)
        self.assertEqual(len(wav), len(wav_processed))
        # Check fundamental armonic movement
        fft_wav = np.fft.fft(wav)
        fft_wav = fft_wav[:int(len(fft_wav) / 2)]
        fft_wav = np.abs(fft_wav)
        fft_wav_proc = np.fft.fft(wav_processed)
        fft_wav_proc = fft_wav_proc[:int(len(fft_wav_proc) / 2)]
        fft_wav_proc = np.abs(fft_wav_proc)
        self.assertLess(np.argmax(fft_wav), np.argmax(fft_wav_proc))
        self.assertEqual(np.max(np.abs(wav)), np.max(np.abs(wav_processed)))  # Check peak

        wav_processed = pitch_shift_wavfile(wav=wav, sr=sample_rate, n_octaves=-0.5)
        self.assertEqual(len(wav), len(wav_processed))
        # Check fundamental armonic movement
        fft_wav = np.fft.fft(wav)
        fft_wav = fft_wav[:int(len(fft_wav) / 2)]
        fft_wav = np.abs(fft_wav)
        fft_wav_proc = np.fft.fft(wav_processed)
        fft_wav_proc = fft_wav_proc[:int(len(fft_wav_proc) / 2)]
        fft_wav_proc = np.abs(fft_wav_proc)
        self.assertGreater(np.argmax(fft_wav), np.argmax(fft_wav_proc))
        self.assertEqual(np.max(np.abs(wav)), np.max(np.abs(wav_processed)))  # Check peak

    def test_randomly_distort_wavfile(self):
        sample_rate, wav = read_wavfile(self.wav_filepath)
        wav_processed = randomly_distort_wavfile(wav=wav, sr=sample_rate)
        self.assertEqual(len(wav), len(wav_processed))

    def test_draw_random_subclip(self):
        sample_rate, wav = read_wavfile(self.wav_filepath)
        subclip = draw_random_subclip(wav, 1000)
        self.assertEqual(1000, len(subclip))
        self.assertLess(0.5, np.max(np.array(subclip) != 0))
