import os

import numpy as np
import RnnModel
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
import random
import torch
import utils
import time
import re

from sklearn.model_selection import train_test_split
from constants import model_params, base_data_path
from datetime import datetime as Datetime


class Trainer(object):
    def __init__(
        self, seed=42, test_p=0.05, use_cuda=True,
        valid_p=0.05
    ):
        self.tensorboard_started = False
        self.tfile_writer = None
        self.vfile_writer = None
        self.date_stamp = self.make_date_stamp()

        self.valid_p = valid_p

        self.use_cuda = use_cuda
        self.test_p = test_p
        self.seed = seed

        self.model = self.make_model()
        self.dirpath = '../dessa-fake-voice/DS_10283_3336/LA/'

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=utils.model_params['learning_rate']
        )

        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to_cuda()
        else:
            self.device = torch.device("cpu")

        self.audio_labels = self.get_labels()
        # print(f'KEYS {self.audio_labels}')
        self.file_paths = list(self.audio_labels.keys())
        self.labels = list(self.audio_labels.values())

        x_train, x_test, y_train, y_test = train_test_split(
            self.file_paths, self.labels,
            test_size=self.test_p, random_state=self.seed
        )

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.train_reals = None
        self.train_fakes = None
        self.test_reals = None
        self.test_fakes = None
        self.segregate_samples()

    def segregate_samples(self):
        self.train_reals = []
        self.train_fakes = []
        self.test_reals = []
        self.test_fakes = []

        for filepath in self.x_train:
            label = self.audio_labels[filepath]

            if label == 1:
                self.train_fakes.append(filepath)
            else:
                self.train_reals.append(filepath)

        for filepath in self.x_test:
            label = self.audio_labels[filepath]

            if label == 1:
                self.test_fakes.append(filepath)
            else:
                self.test_reals.append(filepath)

    def get_labels(self):
        filepath = os.path.join(
            self.dirpath, 'ASVspoof2019_LA_cm_protocols',
            'ASVspoof2019.LA.cm.train.trn.txt'
        )

        label_data = open(filepath).read().strip().split('\n')
        audio_labels = {}

        for line in label_data:
            matches = re.findall('LA_T_[^\\s]*', line)
            assert len(matches) == 1
            filename = matches[0]

            subdir = 'ASVspoof2019_LA_train/flac'
            path = (
                os.path.join(self.dirpath, subdir),
                f'{filename}.flac'
            )

            # print(f'PATH {path}')
            # input('>>> ')
            # path = f'../datasets/train/audios/{filename}'

            if 'bonafide' in line:
                audio_labels[path] = 0
            else:
                assert 'spoof' in line
                audio_labels[path] = 1

        return audio_labels

    def train(
        self, episodes=10 * 1000, batch_size=16,
        fake_p=0.5, target_lengths=(128, 1024)
    ):
        # ASVspoof2019_LA_cm_protocols
        self.tensorboard_start()
        episode_no = 0
        validate_eps = 0
        train_eps = 0

        run_validation = False
        start_time = time.perf_counter()

        while episode_no <= episodes:
            if run_validation:
                print(f'VA episode {episode_no}/{episodes}')
                validate_eps += batch_size
                run_validation = False

                self.batch_validate(
                    episode_no, batch_size=batch_size, fake_p=fake_p,
                    target_lengths=target_lengths
                )
            else:
                print(f'TR episode {episode_no}/{episodes}')
                train_eps += batch_size

                validate_threshold = train_eps * self.valid_p
                run_validation = validate_threshold > validate_eps

                self.batch_train(
                    episode_no, batch_size=batch_size, fake_p=fake_p,
                    target_lengths=target_lengths,
                    record=run_validation
                )

            episode_no += batch_size

        save_path = f'saves/models/{self.date_stamp}.pt'
        torch.save(self.model.state_dict(), save_path)
        print(f'model saved at {save_path}')

        end_time = time.perf_counter()
        time_taken = end_time - start_time
        time_per_episode = time_taken / episode_no
        print(f'time taken: {round(time_taken, 2)}s')
        print(f'time per eps: {round(time_per_episode, 3)}s')

    def batch_validate(
        self, episode_no, batch_size=16, fake_p=0.5,
        target_lengths=(128, 1024)
    ):
        batch_x, np_labels = self.prepare_batch(
            batch_size=batch_size, fake_p=fake_p,
            target_lengths=target_lengths
        )

        torch_batch_x = torch.tensor(batch_x).to(self.device)
        torch_labels = torch.tensor(np_labels).float()
        torch_labels = torch_labels.to(self.device).detach()
        preds, h0 = self.model(torch_batch_x)
        loss = self.criterion(preds, torch_labels)
        loss_value = loss.item()

        np_preds = preds.cpu().detach().numpy().flatten()
        flat_labels = np_labels.flatten()
        self.record_metrics(
            episode_no, loss_value, np_preds, flat_labels,
            callback=self.record_validate_errors
        )

    def batch_train(
        self, episode_no, batch_size=16, fake_p=0.5,
        target_lengths=(128, 1024), record=False
    ):
        batch_x, np_labels = self.prepare_batch(
            batch_size=batch_size, fake_p=fake_p,
            target_lengths=target_lengths
        )

        torch_batch_x = torch.tensor(batch_x).to(self.device)
        torch_labels = torch.tensor(np_labels).float()
        torch_labels = torch_labels.to(self.device).detach()
        preds, h0 = self.model(torch_batch_x)

        self.optimizer.zero_grad()
        loss = self.criterion(preds, torch_labels)
        loss_value = loss.item()
        loss.backward()
        self.optimizer.step()

        if record:
            np_preds = preds.detach().cpu().numpy().flatten()
            flat_labels = np_labels.flatten()
            self.record_metrics(
                episode_no, loss_value, np_preds, flat_labels,
                callback=self.record_train_errors
            )

    @staticmethod
    def record_metrics(
        episode_no, loss_value, np_preds, flat_labels,
        callback
    ):
        round_preds = np.round(np_preds)
        matches = np.equal(round_preds, flat_labels)
        accuracy = sum(matches) / len(matches)

        errors = abs(np_preds - flat_labels)
        me = np.sum(errors) / errors.size
        squared_error = np.sum((np_preds - flat_labels) ** 2)
        mse = (squared_error / errors.size) ** 0.5

        callback(
            step=episode_no, loss=loss_value,
            me=me, mse=mse, accuracy=accuracy
        )

    def prepare_batch(
        self, batch_size=16, fake_p=0.5, target_lengths=(128, 128),
        is_training=True
    ):
        num_fake = int(batch_size * fake_p)
        fake_filepaths = self.get_rand_filepaths(
            1, num_fake, is_training=is_training
        )
        num_real = batch_size - num_fake
        real_filepaths = self.get_rand_filepaths(
            0, num_real, is_training=is_training
        )

        batch_filepaths = fake_filepaths + real_filepaths
        batch_labels = [1] * num_fake + [0] * num_real

        process_batch = utils.preprocess_from_filenames(
            batch_filepaths, '', batch_labels, use_parallel=True
        )

        batch = [episode[0] for episode in process_batch]
        target_length = random.choice(range(
            target_lengths[0], target_lengths[1] + 1
        ))

        min_length = float('inf')
        for audio_arr in batch:
            min_length = min(min_length, len(audio_arr))

        data_batch = []
        min_length = min(min_length, target_length)

        for episode in batch:
            if len(episode) == min_length:
                data_batch.append(episode)
                continue

            max_start = len(episode) - min_length
            start = random.choice(range(max_start))
            clip_episode = episode[start: start + min_length]
            data_batch.append(clip_episode)

        data_batch = np.array(data_batch)
        assert data_batch.shape[2] == utils.hparams.num_mels
        batch_x = data_batch.reshape((
            len(data_batch), -1, utils.hparams.num_mels, 1
        ))

        np_labels = np.array(batch_labels)
        np_labels = np.expand_dims(np_labels, axis=-1)
        return batch_x, np_labels

    def record_validate_errors(
        self, step, loss, me=-1, mse=-1, accuracy=-1
    ):
        assert self.tensorboard_started
        # input(f'VALIDATION ERRORS RECORD')

        with self.vfile_writer.as_default():
            tf.summary.scalar('loss', data=loss, step=step)
            tf.summary.scalar('mean_error', data=me, step=step)
            tf.summary.scalar('accuracy', data=accuracy, step=step)
            tf.summary.scalar(
                'mean_squared_error', data=mse, step=step
            )

            self.vfile_writer.flush()

    def record_train_errors(self, step, loss, me=-1, mse=-1, accuracy=-1):
        # input('PRE-RECORD')
        assert self.tensorboard_started

        with self.tfile_writer.as_default():
            tf.summary.scalar('loss', data=loss, step=step)
            tf.summary.scalar('mean_error', data=me, step=step)
            tf.summary.scalar('accuracy', data=accuracy, step=step)
            tf.summary.scalar(
                'mean_squared_error', data=mse, step=step
            )

            self.tfile_writer.flush()

        # input('AFT-RECORD')

    def get_rand_filepaths(
        self, label=0, episodes=10, is_training=True
    ):
        filepaths = []
        for k in range(episodes):
            filepaths.append(self.get_rand_filepath(
                label, is_training=is_training
            ))

        return filepaths

    def get_rand_filepath(self, label=0, is_training=True):
        assert label in (0, 1)

        if training:
            if label == 0:
                filepath = random.choice(self.train_reals)
            else:
                filepath = random.choice(self.train_fakes)
        else:
            if label == 0:
                filepath = random.choice(self.test_reals)
            else:
                filepath = random.choice(self.test_fakes)

        return filepath

    def fetch_episode(self, label=0):
        filepath = self.get_filepath(label=label)
        array = process_audio_files_inference(
            filepath, '', 'unlabeled'
        )

        return array

    @staticmethod
    def make_date_stamp():
        return Datetime.now().strftime("%y%m%d-%H%M")

    def tensorboard_start(self):
        log_dir = f'saves/logs/AUD-{self.date_stamp}'
        train_path = log_dir + '/training'
        valid_path = log_dir + '/validation'
        self.tfile_writer = tf.summary.create_file_writer(train_path)
        self.vfile_writer = tf.summary.create_file_writer(valid_path)
        self.tensorboard_started = True

    @staticmethod
    def make_model():
        discriminator = RnnModel.Discriminator(
            num_freq_bin=model_params['num_freq_bin'],
            init_neurons=model_params['num_conv_filters'],
            num_conv_blocks=model_params['num_conv_blocks'],
            residual_con=model_params['residual_con'],
            num_dense_neurons=model_params['num_dense_neurons'],
            dense_dropout=model_params['dense_dropout'],
            num_dense_layers=model_params['num_dense_layers'],
            hidden_size=5
        )

        return discriminator


