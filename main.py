import json
import multiprocessing
import os
import random
import sys

import numpy as np
import torch
from tensorboardX import SummaryWriter

from src.architecture import XceptionArchitecture1d
from src.common_paths import get_tensorboard_logs_path, get_model_path, get_dataset_filepath, get_augmented_data_folder
from src.constants import available_tasks, commands, unknown_class_addition
from src.data_processing import get_list_of_wav_paths, batch_augment_files, DataFeeder, download_dataset, \
    decompress_dataset


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


if __name__ == "__main__":
    experiment_settings_filepath = sys.argv[1]  # Retrieve the path of the model_settings.json file
    # Parameters configuration
    task = json.load(open(experiment_settings_filepath))["task"]
    data_version = json.load(open(experiment_settings_filepath))["data_version"]
    model_alias = json.load(open(experiment_settings_filepath))["model_alias"]
    alias = f"{task}_m-{model_alias}_d-{data_version}"
    assert task in available_tasks
    known_commands = commands[task][:]
    include_unknown = unknown_class_addition[task]

    n_jobs = multiprocessing.cpu_count()
    n_epochs = json.load(open(experiment_settings_filepath))["n_epochs"]
    batch_size = json.load(open(experiment_settings_filepath))["batch_size"]
    run_in_gpu = json.load(open(experiment_settings_filepath))["run_in_gpu"]
    n_augmentations = json.load(open(experiment_settings_filepath))["n_augmentations"]
    bn_momentum = json.load(open(experiment_settings_filepath))["bn_momentum"]
    weight_decay = json.load(open(experiment_settings_filepath))["weight_decay"]


    # Download and decompress the data if necessary
    if not os.path.exists(get_dataset_filepath(data_version)):
        download_dataset(data_version)
        decompress_dataset(data_version)

    # Generate data augmentations if required
    random.seed(655321)
    np.random.seed(655321)
    torch.random.manual_seed(655321)

    if n_augmentations > 0:
        train_files, _, _ = get_list_of_wav_paths(data_version=data_version)
        for i in range(n_augmentations):
            if len(os.listdir(get_augmented_data_folder(data_version, str(i)))) == 0:
                print("Generating augmentation no. {}".format(i))
                batch_augment_files(data_version=data_version, list_of_files=train_files, folder_name=str(i),
                                    n_jobs=n_jobs)

    # Load all data paths and build the data feeders
    random.seed(655321)
    np.random.seed(655321)
    torch.random.manual_seed(655321)

    train_paths, validation_paths, test_paths = get_list_of_wav_paths(data_version=data_version,
                                                                      n_augmentations=n_augmentations)
    data_feeder_train = DataFeeder(data_version=data_version, file_paths=train_paths, batch_size=batch_size,
                                   include_silence=False, include_unknown=include_unknown,
                                   known_commands=known_commands[:])
    data_feeder_validation = DataFeeder(data_version=data_version, file_paths=validation_paths, batch_size=batch_size,
                                        include_silence=False, include_unknown=include_unknown,
                                        known_commands=known_commands[:])
    data_feeder_test = DataFeeder(data_version=data_version, file_paths=test_paths, batch_size=batch_size,
                                  include_silence=False, include_unknown=include_unknown,
                                  known_commands=known_commands[:])
   
    assert data_feeder_train.target_encoder == data_feeder_validation.target_encoder == data_feeder_test.target_encoder
    
    assert data_feeder_train.known_commands == data_feeder_validation.known_commands == data_feeder_test.known_commands
    
    # Load architecture
    model = XceptionArchitecture1d(n_classes=len(data_feeder_train.known_commands), lr=1e-4, bn_momentum=bn_momentum,
                                   weight_decay=weight_decay)
    if run_in_gpu: model.cuda()

    # Instantiate summary writer for tensorboard
    sw = SummaryWriter(log_dir=os.path.join(get_tensorboard_logs_path(), alias))
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=model.optimizer,
                                                              mode="max",
                                                              factor=0.5,
                                                              patience=4,
                                                              verbose=True)
    best_score = 0
    c = 0
    loss_train, accuracy_train = None, None
    for epoch in range(n_epochs):
        # Evaluate model
        loss_val, accuracy_val = evaluate_model(model=model, data_feeder=data_feeder_validation, run_in_gpu=run_in_gpu)
        sw.add_scalar("validation/loss", loss_val, c)
        sw.add_scalar('validation/accuracy', accuracy_val, c)

        print(f"[{epoch + 1}] Loss train: {loss_train} | Acc train: {accuracy_train} | Loss val: {loss_val} | Acc val: {accuracy_val}")
        lr_scheduler.step(accuracy_val)

        # Save model
        if accuracy_val > best_score:
            torch.save(model.state_dict(), os.path.join(get_model_path(model_alias=alias), f'checkpoint.pth'))
            best_score = accuracy_val

        # Train model
        loss_train, accuracy_train = 0, 0
        data_feeder_train.shuffle_data() # Shuffle the whole data matrix to get rid of incremental gradient [Bengio 2012]
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

