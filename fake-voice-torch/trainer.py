import bisect
import os

import numpy as np
import model
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
import random
import pandas as pd
import torch
import utils
import time
import math
import re

from sklearn.model_selection import train_test_split
from constants import model_params, base_data_path
from datetime import datetime as Datetime
from tqdm.auto import tqdm

torch.cuda.set_per_process_memory_fraction(0.5, 0)
torch.cuda.empty_cache()

def round_sig(x, sig=2):
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


class Samples(object):
    def __init__(self, filenames):
        self.filenames = filenames
        self.durations = None
        self.cum_weights = None
        self.locked = False

    def append(self, filename):
        assert not self.locked
        self.filenames.append(filename)

    def reorder_by_durations(self, dirpath=''):
        self.locked = True
        self.durations = utils.get_durations(
            filenames=self.filenames, dirpath=dirpath
        )

        self.filenames = np.array(self.filenames)
        self.durations = np.array(self.durations)
        indexes = np.argsort(self.durations)

        print('TOTAL DURATION', sum(self.durations))

        self.filenames = self.filenames[indexes]
        self.durations = self.durations[indexes]
        cum_durations = np.cumsum(self.durations)
        self.cum_weights = cum_durations / sum(self.durations)
        assert max(self.cum_weights) <= 1

    def weighted_sample(self):
        number = random.random()
        index = bisect.bisect_left(self.cum_weights, number)
        return self.filenames[index]

    def random_sample(self):
        return random.choice(self.filenames)

    def sample(self):
        if self.cum_weights is None:
            return self.random_sample()
        else:
            return self.weighted_sample()


class Trainer(object):
    def __init__(
        self, seed=42, test_p=0.1, use_cuda=True,
        valid_p=0.01, weigh_sampling=True, add_aisg=True
    ):
        self.tensorboard_started = False
        self.tfile_writer = None
        self.vfile_writer = None
        self.date_stamp = self.make_date_stamp()

        self.accum_train_score = 0
        self.accum_validate_score = 0
        self.weigh_sampling = weigh_sampling
        self.save_best_every = 1000
        self.perf_decay = 0.96

        self.valid_p = valid_p

        self.use_cuda = use_cuda
        self.test_p = test_p
        self.seed = seed

        self.model = self.make_model()
        self.dirpath = '../dessa-fake-voice/DS_10283_3336/LA/'
        self.lr = model_params['learning_rate']
        # self.lr = 0.001

        self.criterion = nn.BCELoss()
        self.params = self.model.load_parameters()
        self.optimizer = optim.Adam(self.params, lr=self.lr)

        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to_cuda()
        else:
            self.device = torch.device("cpu")

        self.add_aisg = add_aisg
        self.audio_labels = self.get_labels(add_aisg)
        # print(f'KEYS {self.audio_labels}')
        self.labels = list(self.audio_labels.values())
        self.file_paths = list(self.audio_labels.keys())

        x_train, x_test, y_train, y_test = train_test_split(
            self.file_paths, self.labels,
            test_size=self.test_p, random_state=self.seed
        )

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.samples = None
        self.segregate_samples()

    @property
    def train_reals(self):
        return self.samples['train_real']

    @property
    def train_fakes(self):
        return self.samples['train_fake']

    @property
    def test_reals(self):
        return self.samples['test_real']

    @property
    def test_fakes(self):
        return self.samples['test_fake']

    def segregate_samples(self):
        train_reals = Samples([])
        train_fakes = Samples([])
        test_reals = Samples([])
        test_fakes = Samples([])

        for filepath in self.x_train:
            label = self.audio_labels[filepath]

            if label == 1:
                train_fakes.append(filepath)
            else:
                train_reals.append(filepath)

        for filepath in self.x_test:
            label = self.audio_labels[filepath]

            if label == 1:
                test_fakes.append(filepath)
            else:
                test_reals.append(filepath)

        self.samples = {
            'train_real': train_reals, 'test_real': test_reals,
            'train_fake': train_fakes, 'test_fake': test_fakes
        }

        if self.weigh_sampling:
            for name, group_samples in self.samples.items():
                print('REORDERING', name)
                group_samples.reorder_by_durations()

    def get_labels(self, add_aisg=True):
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

        if add_aisg:
            self.load_aisg(audio_labels)

        return audio_labels

    @staticmethod
    def load_aisg(audio_labels):
        # load real audios from AISG dataset
        df = pd.read_csv('../datasets/extra-labels.csv')
        filenames = df[df['label'] == 0]['filename'].to_numpy()

        for filename in filenames:
            name = filename[:filename.index('.')]
            file_path = f'../datasets/train/audios-flac/{name}.flac'
            audio_labels[file_path] = 0

        print(f'LOADED {len(filenames)} AISG REAL FILES')
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
        best_score = float('-inf')
        save_folder = f'saves/checkpoints/{self.date_stamp}'
        pbar = tqdm(range(episodes))

        last_checkpoint = 0
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        while episode_no <= episodes:
            if run_validation:
                desc = f'VA episode {episode_no}/{episodes}'
                validate_eps += batch_size
                run_validation = False

                score = self.batch_validate(
                    episode_no, batch_size=batch_size, fake_p=fake_p,
                    target_lengths=target_lengths
                )

                self.accum_validate(score, self.perf_decay)
            else:
                desc = f'TR episode {episode_no}/{episodes}'
                train_eps += batch_size

                validate_threshold = train_eps * self.valid_p
                run_validation = validate_threshold > validate_eps

                score = self.batch_train(
                    episode_no, batch_size=batch_size, fake_p=fake_p,
                    target_lengths=target_lengths,
                    record=run_validation
                )

                self.accum_train(score, self.perf_decay)
                
            pbar.set_description(desc)
            pbar.update(batch_size)
            episode_no += batch_size
            time.sleep(0.1)

            # at_checkpoint = episode_no % self.save_best_every == 0
            episodes_past = episode_no - last_checkpoint
            at_checkpoint = episodes_past > self.save_best_every

            if (episode_no > 0) and at_checkpoint:
                last_checkpoint = episode_no
                best_score = self.update_best_model(
                    validate_eps=validate_eps, best_score=best_score,
                    decay=self.perf_decay, train_eps=train_eps,
                    save_folder=save_folder
                )

        save_path = f'saves/models/{self.date_stamp}.pt'
        torch.save(self.model.state_dict(), save_path)
        print(f'model saved at {save_path}')

        end_time = time.perf_counter()
        time_taken = end_time - start_time
        time_per_episode = time_taken / episode_no
        print(f'time taken: {round(time_taken, 2)}s')
        print(f'time per eps: {round(time_per_episode, 3)}s')

    @staticmethod
    def get_smooth_score(accum_score, decay, eps):
        denom = (1 - decay ** eps) / (1 - decay)
        score = accum_score / denom
        return score

    def accum_train(self, reward, decay):
        self.accum_train_score += reward
        self.accum_train_score *= decay
        return self.accum_train_score

    def accum_validate(self, reward, decay):
        self.accum_validate_score += reward
        self.accum_validate_score *= decay
        return self.accum_validate_score

    def update_best_model(
        self, decay, train_eps, validate_eps, best_score, save_folder
    ):
        train_score = self.get_smooth_score(
            self.accum_train_score, decay, train_eps
        )
        validate_score = self.get_smooth_score(
            self.accum_validate_score, decay, validate_eps
        )

        episodes = train_eps + validate_eps
        smooth_score = min(train_score, validate_score)
        round_train = round_sig(train_score, sig=2)
        round_validate = round_sig(validate_score, sig=2)

        if round_train == int(round_train):
            round_train = int(round_train)
        if round_validate == int(round_validate):
            round_validate = int(round_validate)

        print(f'TRAIN ~ {round_train} VALIDATE ~ {round_validate}')
        name = f'E{episodes}_T{round_train}_V{round_validate}'
        save_path = f'{save_folder}/{name}.pt'

        if smooth_score > best_score:
            best_score = smooth_score
            torch.save(self.model.state_dict(), save_path)

            print(f'SAVING BEST MODEL @ EPS {episodes}')
            print(f'BEST SAVED AT {save_path}')
        else:
            print(f'NEGLECTING BEST MODEL @ EPS {episodes}')

        return best_score

    def batch_validate(
        self, episode_no, batch_size=16, fake_p=0.5,
        target_lengths=(128, 1024)
    ):
        batch_x, np_labels = self.prepare_batch(
            batch_size=batch_size, fake_p=fake_p,
            target_lengths=target_lengths,
            is_training=False
        )

        torch_batch_x = torch.tensor(batch_x).to(self.device)
        torch_labels = torch.tensor(np_labels).float()
        torch_labels = torch_labels.to(self.device).detach()

        # self.optimizer.zero_grad()
        preds = self.model(torch_batch_x)
        loss = self.criterion(preds, torch_labels)
        loss_value = loss.item()

        np_preds = preds.detach().cpu().numpy().flatten()
        flat_labels = np_labels.flatten()
        score = self.record_metrics(
            episode_no, loss_value, np_preds, flat_labels,
            callback=self.record_validate_errors
        )

        return score

    def batch_train(
        self, episode_no, batch_size=16, fake_p=0.5,
        target_lengths=(128, 1024), record=False
    ):
        batch_x, np_labels = self.prepare_batch(
            batch_size=batch_size, fake_p=fake_p,
            target_lengths=target_lengths,
            is_training=True
        )

        torch_batch_x = torch.tensor(batch_x).to(self.device)
        torch_labels = torch.tensor(np_labels).float()
        torch_labels = torch_labels.to(self.device).detach()
        preds = self.model(torch_batch_x)

        self.optimizer.zero_grad()
        loss = self.criterion(preds, torch_labels)
        loss_value = loss.item()
        loss.backward()
        self.optimizer.step()

        callback = self.record_train_errors if record else None

        np_preds = preds.detach().cpu().numpy().flatten()
        flat_labels = np_labels.flatten()
        score = self.record_metrics(
            episode_no, loss_value, np_preds, flat_labels,
            callback=callback
        )

        return score

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

        if callback is not None:
            callback(
                step=episode_no, loss=loss_value,
                me=me, mse=mse, accuracy=accuracy
            )

        score = 2 * accuracy - 1
        return score

    def prepare_batch(
        self, batch_size=16, fake_p=0.5, target_lengths=(128, 128),
        is_training=True, mel_batch_size=16
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
            batch_filepaths, '', batch_labels, use_parallel=True,
            num_cores=8, show_pbar=False
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

        np_labels = np.array([
            np.ones((min_length, 1)) * label
            for label in batch_labels
        ])
        # np_labels = np.expand_dims(np_labels, axis=-1)
        return batch_x, np_labels

    def record_validate_errors(self, *args, **kwargs):
        self.record_errors(
            file_writer=self.vfile_writer, *args, **kwargs
        )

    def record_train_errors(self, *args, **kwargs):
        self.record_errors(
            file_writer=self.tfile_writer, *args, **kwargs
        )

    def record_errors(
        self, file_writer, step, loss, me=-1, mse=-1,
        accuracy=-1
    ):
        assert self.tensorboard_started
        score = 2 * accuracy - 1

        with file_writer.as_default():
            tf.summary.scalar('loss', data=loss, step=step)
            tf.summary.scalar('mean_error', data=me, step=step)
            tf.summary.scalar('accuracy', data=accuracy, step=step)
            tf.summary.scalar('score', data=score, step=step)
            tf.summary.scalar(
                'mean_squared_error', data=mse, step=step
            )

            file_writer.flush()

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
                filepath = self.train_reals.sample()
            else:
                filepath = self.train_fakes.sample()
        else:
            if label == 0:
                filepath = self.test_reals.sample()
            else:
                filepath = self.test_fakes.sample()

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
        discriminator = model.Discriminator(
            num_freq_bin=model_params['num_freq_bin'],
            init_neurons=model_params['num_conv_filters'],
            num_conv_blocks=model_params['num_conv_blocks'],
            residual_con=model_params['residual_con'],
            num_dense_neurons=model_params['num_dense_neurons'],
            dense_dropout=model_params['dense_dropout'],
            num_dense_layers=model_params['num_dense_layers'],
            spatial_dropout_fraction=model_params[
                'spatial_dropout_fraction'
            ]
        )

        return discriminator


