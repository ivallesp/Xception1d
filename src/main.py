import json
import os

from src.common_paths import get_training_data_path
from src.data_processing import get_list_of_wav_paths, batch_augment_files

# Generate data augmentations
n_augmentations = json.load(open("settings.json"))["n_augmentations"]
if n_augmentations > 0:
    tr, va, te = get_list_of_wav_paths()
    tr = [os.path.join(get_training_data_path(), "audio", f) for f in tr]
    batch_augment_files(tr, n_augmentations, n_jobs=4)
