import json
import multiprocessing
import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from src.architecture import XceptionArchitecture1d
from src.common_paths import get_tensorboard_logs_path, get_model_path
from src.data_processing import get_list_of_wav_paths, batch_augment_files, DataFeeder

random.seed(655321)
np.random.seed(655321)
torch.random.manual_seed(655321)

n_jobs = multiprocessing.cpu_count()
n_epochs = json.load(open("settings.json"))["n_epochs"]
batch_size = json.load(open("settings.json"))["batch_size"]
run_in_gpu = json.load(open("settings.json"))["run_in_gpu"]
version_name = json.load(open("settings.json"))["version_id"]

# Generate data augmentations
n_augmentations = json.load(open("settings.json"))["n_augmentations"]
if n_augmentations > 0:
    train_files, _, _ = get_list_of_wav_paths()
    batch_augment_files(train_files, n_augmentations, n_jobs=n_jobs)

random.seed(655321)
np.random.seed(655321)
torch.random.manual_seed(655321)
# Load all data paths and build the data feeders
tr, va, te = get_list_of_wav_paths(include_augmentations=True)
data_feeder_train = DataFeeder(tr, batch_size=batch_size, add_noise=True)
data_feeder_validation = DataFeeder(va, batch_size=batch_size, add_noise=True)
data_feeder_test = DataFeeder(te, batch_size=batch_size, add_noise=True)

# Load architecture
model = XceptionArchitecture1d(n_classes=len(data_feeder_train.known_classes), lr=2.5e-4)
if run_in_gpu:
    model.cuda()
# Instantiate summary writer for tensorboard
sw = SummaryWriter(log_dir=os.path.join(get_tensorboard_logs_path(), version_name))
best_score = 0
c = 0

for epoch in range(n_epochs):
    # Evaluate model
    loss_val, accuracy_val = 0, 0
    model.eval()
    for n, (batch_audio_val, batch_target_val) in tqdm(enumerate(data_feeder_validation.get_batches())):
        if run_in_gpu:
            batch_audio_val = batch_audio_val.cuda()
            batch_target_val = batch_target_val.cuda()
        loss, y_hat = model.calculate_loss(batch_audio_val, batch_target_val)
        loss_val += loss.cpu().detach().numpy()
        accuracy_val += (batch_target_val.cpu().numpy() == y_hat.argmax(dim=1).cpu().numpy()).mean()
    model.train()
    print("\n")
    loss_val /= (n + 1)
    accuracy_val /= (n + 1)
    sw.add_scalar("validation/loss", loss_val, c)
    sw.add_scalar('validation/accuracy', accuracy_val, c)
    if accuracy_val > best_score:
        print("Found best score. Saving model...")
        torch.save(model.state_dict(), os.path.join(get_model_path(version_id=version_name), f'checkpoint.pth'))
        print("Model saved successfully!")
        best_score = accuracy_val

    # Train model
    loss_train, accuracy_train = 0, 0
    for n, (batch_audio_train, batch_target_train) in tqdm(enumerate(data_feeder_train.get_batches())):
        if run_in_gpu:
            batch_audio_train = batch_audio_train.cuda()
            batch_target_train = batch_target_train.cuda()
        loss, y_hat = model.step(batch_audio_train, batch_target_train)
        loss_train += loss.detach().cpu().numpy()
        accuracy = (batch_target_train.cpu().numpy() == y_hat.argmax(dim=1).cpu().numpy()).mean()
        accuracy_train += accuracy
        # Load it in TensorboardX
        sw.add_scalar("train/loss", loss, c)
        sw.add_scalar('train/accuracy', accuracy, c)
        c += 1
    print("\n")
    loss_train /= (n + 1)
    accuracy_train /= (n + 1)

    print(
        f"[{epoch + 1}] Loss train: {loss_train} | Acc train: {accuracy_train} | Loss val: {loss_val} | Acc val: {accuracy_val}")
