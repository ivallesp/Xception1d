import json
import multiprocessing
import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter

from src.architecture import XceptionArchitecture1d
from src.common_paths import get_tensorboard_logs_path, get_model_path
from src.data_processing import get_list_of_wav_paths, batch_augment_files, DataFeeder

n_jobs = multiprocessing.cpu_count()
n_epochs = json.load(open("settings.json"))["n_epochs"]
batch_size = json.load(open("settings.json"))["batch_size"]
run_in_gpu = json.load(open("settings.json"))["run_in_gpu"]
version_name = json.load(open("settings.json"))["version_id"]
n_augmentations = json.load(open("settings.json"))["n_augmentations"]

# Generate data augmentations
random.seed(655321)
np.random.seed(655321)
torch.random.manual_seed(655321)

if n_augmentations > 0:
    train_files, _, _, _ = get_list_of_wav_paths()
    batch_augment_files(train_files, n_augmentations, n_jobs=n_jobs)

# Load all data paths and build the data feeders
random.seed(655321)
np.random.seed(655321)
torch.random.manual_seed(655321)

train_paths, validation_paths, test_paths, scoring_paths = get_list_of_wav_paths(include_augmentations=True)
data_feeder_train = DataFeeder(train_paths, batch_size=batch_size, add_noise=True)
data_feeder_validation = DataFeeder(validation_paths, batch_size=batch_size, add_noise=True)
data_feeder_test = DataFeeder(test_paths, batch_size=batch_size, add_noise=True)
data_feeder_scoring = DataFeeder(scoring_paths, batch_size=batch_size, add_noise=False, shuffle=False, scoring=True)

# Load architecture
model = XceptionArchitecture1d(n_classes=len(data_feeder_train.known_classes), lr=1e-4)
if run_in_gpu: model.cuda()
# Instantiate summary writer for tensorboard
sw = SummaryWriter(log_dir=os.path.join(get_tensorboard_logs_path(), version_name))
best_score = 0
c = 0


def evaluate_model(model, data_feeder, run_in_gpu):
    loss_final, accuracy_final = 0, 0
    model.eval()
    for n, (batch_audio_val, batch_target_val) in enumerate(data_feeder.get_batches()):
        if run_in_gpu:
            batch_audio_val = batch_audio_val.cuda()
            batch_target_val = batch_target_val.cuda()
        loss, y_hat = model.calculate_loss(batch_audio_val, batch_target_val)
        loss_final += loss.cpu().detach().numpy()
        accuracy_final += (batch_target_val.cpu().numpy() == y_hat.argmax(dim=1).cpu().numpy()).mean()

    loss_final /= (n + 1)
    accuracy_final /= (n + 1)
    model.train()
    return loss_final, accuracy_final


def model_predict(model, data_feeder, run_in_gpu):
    assert not data_feeder.shuffle
    model.eval()
    preds = []
    for n, (batch_audio, _) in enumerate(data_feeder.get_batches(return_incomplete_batches=True)):
        if run_in_gpu:
            batch_audio = batch_audio.cuda()
        y_hat = model.forward(batch_audio).cpu().detach().numpy()
        preds += [y_hat]
    model.train()
    return np.concatenate(preds)


for epoch in range(n_epochs):
    # Evaluate model
    loss_val, accuracy_val = evaluate_model(model=model, data_feeder=data_feeder_validation, run_in_gpu=run_in_gpu)
    sw.add_scalar("validation/loss", loss_val, c)
    sw.add_scalar('validation/accuracy', accuracy_val, c)

    # Save model
    if accuracy_val > best_score:
        torch.save(model.state_dict(), os.path.join(get_model_path(version_id=version_name), f'checkpoint.pth'))
        best_score = accuracy_val

    # Train model
    loss_train, accuracy_train = 0, 0
    for n, (batch_audio_train, batch_target_train) in enumerate(data_feeder_train.get_batches()):
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
    loss_train /= (n + 1)
    accuracy_train /= (n + 1)

    print(
        f"[{epoch + 1}] Loss train: {loss_train} | Acc train: {accuracy_train} | Loss val: {loss_val} | Acc val: {accuracy_val}")
