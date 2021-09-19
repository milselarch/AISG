import os

import numpy as np
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
import random
import torch
import utils
import time
import audio
import re

import librosa.display
import librosa.filters
import librosa

from sklearn.model_selection import train_test_split
from constants import model_params, base_data_path
from datetime import datetime as Datetime
from wav_models import wav_model
from pdb import set_trace as bp

SET_BP = False

def set_breakpoint():
    if SET_BP:
        bp()

class Trainer(object):
    def __init__(
        self, seed=42, test_p=0.2, use_cuda=True,
        valid_p=0.01
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
        self.wav_model_file = '../wav2lip/pretrained/wav2lip.pth'
        self.bootstrap_model()

        self.dirpath = '../dessa-fake-voice/DS_10283_3336/LA/'
        # self.lr = utils.model_params['learning_rate']
        self.lr = 0.001

        print(f'USING LEARNING RATE {self.lr}')

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr
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

    def bootstrap_model(self):
        checkpoint = torch.load(self.wav_model_file)
        state = checkpoint["state_dict"]
        new_state = {}

        for k, v in state.items():
            new_state[k.replace('module.', '')] = v

        self.model.wav2lip.load_state_dict(new_state)

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
        fake_p=0.5, mel_batch_size=16
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
                    mel_batch_size=mel_batch_size
                )
            else:
                print(f'TR episode {episode_no}/{episodes}')
                train_eps += batch_size

                validate_threshold = train_eps * self.valid_p
                run_validation = validate_threshold > validate_eps

                self.batch_train(
                    episode_no, batch_size=batch_size, fake_p=fake_p,
                    mel_batch_size=mel_batch_size,
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
        mel_batch_size=16
    ):
        batch_x, np_labels = self.prepare_batch(
            batch_size=batch_size, fake_p=fake_p,
            mel_batch_size=mel_batch_size,
            is_training=False
        )

        torch_batch_x = torch.FloatTensor(batch_x).to(self.device)
        torch_labels = torch.FloatTensor(np_labels).float()
        torch_labels = torch_labels.to(self.device).detach()
        self.model.eval()

        with torch.no_grad():
            preds = self.model(torch_batch_x)
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
        mel_batch_size=16, record=False
    ):
        self.model.train()

        batch_x, np_labels = self.prepare_batch(
            batch_size=batch_size, fake_p=fake_p,
            mel_batch_size=mel_batch_size,
            is_training=True
        )

        torch_batch_x = torch.FloatTensor(batch_x).to(self.device)
        torch_labels = torch.FloatTensor(np_labels).float()
        torch_labels = torch_labels.to(self.device).detach()
        preds = self.model(torch_batch_x)

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

    @staticmethod
    def to_mel_chunks(mel, fps, mel_step_size=16):
        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        i = 0

        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break

            mel_chunk = mel[:, start_idx: start_idx + mel_step_size]
            mel_chunks.append(mel_chunk)
            i += 1

        return mel_chunks

    @classmethod
    def random_sample_mels(
        cls, mels, mel_batch_size=16, transpose=True, fps=24
    ):
        mel_chunks = cls.to_mel_chunks(mels, fps=fps)
        index_range = range(0, len(mel_chunks) - mel_batch_size)

        try:
            index = random.choice(index_range)
        except IndexError as e:
            while mel_batch_size != len(mel_chunks):
                pad_size = mel_batch_size - len(mel_chunks)
                pad_size = min(pad_size, len(mel_chunks))
                pad = random.sample(mel_chunks, pad_size)
                mel_chunks.extend(pad)

            index = 0

        sub_mel_batch = mel_chunks[index: index + mel_batch_size]
        np_sub_mel_batch = np.asarray(sub_mel_batch)
        reshape_mel_batch = np.reshape(np_sub_mel_batch, [
            len(np_sub_mel_batch), np_sub_mel_batch.shape[1],
            np_sub_mel_batch.shape[2], 1
        ])

        if transpose:
            reshape_mel_batch = np.transpose(
                reshape_mel_batch, (0, 3, 1, 2)
            )
        return reshape_mel_batch

    def prepare_batch(
        self, batch_size=16, fake_p=0.5, mel_batch_size=16,
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
        prep_labels = [1] * num_fake + [0] * num_real
        process_batch = utils.preprocess_from_filenames(
            batch_filepaths, '', prep_labels, use_parallel=True,
            func=self.load_wav
        )

        batch = [episode[0] for episode in process_batch]
        batch_labels = [episode[1] for episode in process_batch]
        assert batch_labels == prep_labels

        data_batch = []
        for episode in batch:
            clip_episode = self.random_sample_mels(
                episode, mel_batch_size=mel_batch_size
            )
            data_batch.append(clip_episode)

        np_labels = np.concatenate([
            np.ones((mel_batch_size, 1)) * label
            for label in batch_labels
        ])

        print('DATA BATCH')
        # set_breakpoint()
        return data_batch, np_labels

    @staticmethod
    def load_wav(filename, dirpath, file_mode):
        if type(filename) is tuple:
            filename = os.path.join(*filename)

        path = os.path.join(dirpath, filename)
        wav = audio.load_wav(path, sr=16000)
        # mel = librosa.effects.trim(wav)
        mel = audio.melspectrogram(wav)
        mel, index = librosa.effects.trim(mel)

        assert file_mode in (0, 1)
        return mel, file_mode

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

    def record_train_errors(
        self, step, loss, me=-1, mse=-1, accuracy=-1
    ):
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

        if is_training:
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
        discriminator = wav_model.WavDisc()
        return discriminator


