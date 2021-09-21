import bisect
import os

import model
import numpy as np
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
import torch.multiprocessing as mp
import pandas as pd
import random
import torch
import utils
import time
import math
import re

from datetime import timedelta
from sklearn.model_selection import train_test_split
from torch.multiprocessing import Queue, Process, set_start_method
from constants import model_params, base_data_path
from datetime import datetime as Datetime
from tqdm.auto import tqdm


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

        duration = int(sum(self.durations))
        duration_obj = timedelta(seconds=duration)
        print('TOTAL DURATION', duration_obj)

        self.filenames = self.filenames[indexes]
        self.durations = self.durations[indexes]
        cum_durations = np.cumsum(self.durations)
        self.cum_weights = cum_durations / sum(self.durations)
        assert max(self.cum_weights) <= 1

    def weighted_sample(self):
        start = time.perf_counter()
        number = random.random()
        index = bisect.bisect_left(self.cum_weights, number)
        end = time.perf_counter()
        duration = end - start
        return self.filenames[index]

    def random_sample(self):
        return random.choice(self.filenames)

    def sample(self):
        if self.cum_weights is None:
            return self.random_sample()
        else:
            return self.weighted_sample()

    def get_samples(self, n):
        samples = [self.sample() for k in range(n)]
        return samples
    
    
class BaseTrainer(object):
    def __init__(self):
        self.samples = None
        self.tensorboard_started = False
        self.tfile_writer = None
        self.vfile_writer = None

    def profile_train(self):
        raise NotImplemented

    def train(self):
        raise NotImplemented

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