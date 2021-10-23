import bisect
import copy
import os

import model
import numpy as np
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
import torch.multiprocessing as mp
import pandas as pd
import Dataset
import random
import torch
import utils
import time
import math
import re

from datetime import timedelta
from BaseTrainer import BaseTrainer
from sklearn.model_selection import train_test_split
from torch.multiprocessing import Queue, Process, set_start_method
from constants import model_params, base_data_path
from datetime import datetime as Datetime
from tqdm.auto import tqdm

mp.set_start_method('spawn')
# torch.cuda.set_per_process_memory_fraction(0.5, 0)
torch.cuda.empty_cache()

def round_sig(x, sig=2):
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


class Trainer(BaseTrainer):
    def __init__(
        self, seed=42, test_p=0.1, use_cuda=True,
        valid_p=0.05, weigh_sampling=True, add_aisg=True,
        cache_threshold=20, load_dataset=True, save_threshold=0.01,
        use_batch_norm=True, params=None, use_avs=False,
        train_version=1, normalize_audio=False,
        preload_path=None
    ):
        super().__init__()
        self.date_stamp = self.make_date_stamp()

        self.accum_train_score = 0
        self.accum_validate_score = 0
        self.weigh_sampling = weigh_sampling
        self.save_threshold = save_threshold
        self.save_best_every = 2500
        self.perf_decay = 0.96

        self.valid_p = valid_p
        self.cache_threshold = cache_threshold
        self.train_version = train_version
        self.cache = {}

        self.normalize_audio = normalize_audio
        self.use_batch_norm = use_batch_norm

        self.use_cuda = use_cuda
        self.test_p = test_p
        self.seed = seed

        self.params = params
        self.model = self.make_model(params)
        self.dirpath = '../dessa-fake-voice/DS_10283_3336/LA/'
        self.lr = model_params['learning_rate']
        # self.lr = 0.001

        self.criterion = nn.BCELoss()
        self.params = tuple(self.model.parameters())
        self.optimizer = optim.Adam(self.params, lr=self.lr)

        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.cuda()
        else:
            self.device = torch.device("cpu")

        self.add_aisg = add_aisg
        self.use_avs = use_avs

        self.audio_labels = None
        self.labels = None
        self.file_paths = None

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        if load_dataset:
            self.load_dataset(self.add_aisg)

        self.preload_path = preload_path
        if preload_path is not None:
            self.load_model(preload_path)

    def load_model(self, model_path, eval_mode=True):
        self.model.load_state_dict(torch.load(model_path))
        if eval_mode:
            self.model.eval()

    def load_dataset(self, add_aisg):
        self.audio_labels = {}

        if self.use_avs:
            self.audio_labels = self.get_labels()

        if add_aisg:
            self.load_aisg2()

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

        self.segregate_samples()

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
            path = os.path.join(
                self.dirpath, subdir, f'{filename}.flac'
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

    def load_aisg2(self):
        # load real audios from AISG dataset
        df = pd.read_csv('csvs/unique-audios-210930-2218.csv')
        real_cond = df['fake_audio'] == 0
        fake_cond = df['fake_audio'] == 1

        real_filenames = df[real_cond]['filename'].to_numpy()
        fake_filenames = df[fake_cond]['filename'].to_numpy()
        filenames = np.concatenate([real_filenames, fake_filenames])

        for filename in filenames:
            name = filename[:filename.index('.')]
            file_path = f'../datasets-local/audios-flac/{name}.flac'
            label = 0 if filename in real_filenames else 1
            self.audio_labels[file_path] = label

        print(f'LOADED {len(real_filenames)} AISG REAL FILES')
        print(f'LOADED {len(fake_filenames)} AISG FAKE FILES')

    @staticmethod
    def load_aisg(audio_labels):
        # load real audios from AISG dataset
        df = pd.read_csv('../datasets/extra-labels.csv')
        filenames = df[df['label'] == 0]['filename'].to_numpy()

        for filename in filenames:
            name = filename[:filename.index('.')]
            file_path = f'../datasets-local/audios-flac/{name}.flac'
            audio_labels[file_path] = 0

        print(f'LOADED {len(filenames)} AISG REAL FILES')
        return audio_labels

    def train(
        self, episodes=10 * 1000, batch_size=16,
        fake_p=0.5, target_lengths=(128, 256)
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

                if run_validation:
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
        episodes = train_eps + validate_eps

        train_score = self.get_smooth_score(
            self.accum_train_score, decay, episodes
        )
        validate_score = self.get_smooth_score(
            self.accum_validate_score, decay, episodes
        )

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

        if smooth_score - best_score > self.save_threshold:
            best_score = smooth_score
            torch.save(self.model.state_dict(), save_path)

            print(f'SAVING BEST MODEL @ EPS {episodes}')
            print(f'BEST SAVED AT {save_path}')
        else:
            print(f'NEGLECTING BEST MODEL @ EPS {episodes}')

        return best_score

    def batch_train(self, *args, **kwargs):
        if self.train_version == 1:
            return self.batch_train_v1(*args, **kwargs)
        elif self.train_version == 2:
            return self.batch_train_v2(*args, **kwargs)

        raise ValueError(f'BAD TRAIN VERSION {self.train_version}')

    def batch_train_v2(
        self, episode_no, batch_size=16, fake_p=0.5
    ):
        self.model.train()
        batch_x, np_labels = self.prepare_batch(
            batch_size=batch_size, fake_p=fake_p,
            is_training=True, randomize=True
        )

        torch_batch_x = torch.tensor(batch_x).to(self.device)
        torch_labels = torch.tensor(np_labels).float()
        torch_labels = torch_labels.to(self.device).detach()

        total_loss = 0
        self.optimizer.zero_grad()
        batch_outputs = []

        for k, image_input in enumerate(torch_batch_x):
            episode_labels = torch_labels[k]
            episode_preds = self.model.forward_image(image_input)
            loss = self.criterion(episode_preds, episode_labels)
            loss_value = loss.item()
            loss.backward()

            total_loss += loss_value / batch_size
            batch_preds = torch.unsqueeze(episode_preds, 0)
            batch_outputs.append(batch_preds)

        self.optimizer.step()
        callback = self.record_train_errors if record else None
        batch_outputs = torch.cat(batch_outputs, dim=0)

        flat_labels = np_labels.flatten()
        np_preds = batch_outputs.detach().cpu().numpy().flatten()
        score = self.record_metrics(
            episode_no, total_loss, np_preds, flat_labels,
            callback=callback
        )

        return score

    def batch_train_v1(
        self, episode_no, batch_size=16, fake_p=0.5,
        record=False
    ):
        self.model.train()
        batch_x, np_labels = self.prepare_batch(
            batch_size=batch_size, fake_p=fake_p,
            is_training=True, randomize=False
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

    def batch_validate(
        self, episode_no, batch_size=16, fake_p=0.5,
        target_lengths=(128, 1024)
    ):
        batch_x, np_labels = self.prepare_batch(
            batch_size=batch_size, fake_p=fake_p,
            is_training=False, randomize=False
        )

        torch_batch_x = torch.tensor(batch_x).to(self.device)
        torch_labels = torch.tensor(np_labels).float()
        torch_labels = torch_labels.to(self.device).detach()

        # self.optimizer.zero_grad()
        self.model.train(False)

        with torch.no_grad():
            preds = self.model(torch_batch_x)
            loss = self.criterion(preds, torch_labels)
            loss_value = loss.item()

        self.model.train(True)
        np_preds = preds.detach().cpu().numpy().flatten()
        flat_labels = np_labels.flatten()
        score = self.record_metrics(
            episode_no, loss_value, np_preds, flat_labels,
            callback=self.record_validate_errors
        )

        return score

    def batch_predict(self, batch_x, to_numpy=True):
        if type(batch_x) is str:
            batch_x = [batch_x]

        if type(batch_x) in (list, tuple):
            assert type(batch_x[0]) is str
            labels = [1] * len(batch_x)
            batch_x, np_labels = self.load_batch(
                batch_x, labels, cache=False
            )

        if type(batch_x) is np.ndarray:
            torch_batch_x = torch.tensor(batch_x).to(self.device)
        else:
            assert type(batch_x) is torch.Tensor
            torch_batch_x = batch_x

        self.model.eval()
        preds = self.model(torch_batch_x)
        if to_numpy:
            preds = preds.detach().cpu().numpy()

        return preds

    def prepare_batch(
        self, batch_size=16, fake_p=0.5,
        is_training=True, randomize=False
    ):
        # why does randomizing filenames not work?
        # start = time.perf_counter()
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

        if randomize:
            new_batch_labels = []
            new_batch_filepaths = []
            indices = list(range(len(batch_filepaths)))
            random.shuffle(indices)

            for index in indices:
                new_batch_filepaths.append(batch_filepaths[index])
                new_batch_labels.append(batch_labels[index])

            batch_filepaths = new_batch_filepaths
            batch_labels = new_batch_labels

        batch_x, np_labels = self.load_batch(
            batch_filepaths, batch_labels, target_lengths
        )

        assert type(batch_x) is np.ndarray
        assert type(np_labels) is np.ndarray
        return batch_x, np_labels

    def load_batch(
        self, batch_filepaths, batch_labels, target_lengths=None,
        cache=True
    ):
        if cache is True:
            cache = self.cache
        elif cache is False:
            cache = {}

        process_batch = utils.preprocess_from_filenames(
            batch_filepaths, '', batch_labels, use_parallel=True,
            num_cores=4, show_pbar=False, cache=cache,
            cache_threshold=self.cache_threshold,
            normalize=self.normalize_audio
        )

        batch = [episode[0] for episode in process_batch]
        if target_lengths is not None:
            target_length = random.choice(range(
                target_lengths[0], target_lengths[1] + 1
            ))
        else:
            target_length = float('inf')

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
        # end = time.perf_counter()
        # duration = end - start
        return batch_x, np_labels

    def make_model(self, params=None):
        if params is None:
            params = model_params

        discriminator = model.Discriminator(
            bn_momentum=0.99,
            use_batch_norm=self.use_batch_norm,
            num_freq_bin=params['num_freq_bin'],
            init_neurons=params['num_conv_filters'],
            num_conv_blocks=params['num_conv_blocks'],
            residual_con=params['residual_con'],
            num_dense_neurons=params['num_dense_neurons'],
            dense_dropout=params['dense_dropout'],
            num_dense_layers=params['num_dense_layers'],
            spatial_dropout_fraction=params[
                'spatial_dropout_fraction'
            ]
        )

        return discriminator


