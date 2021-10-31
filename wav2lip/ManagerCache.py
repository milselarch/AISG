import time
import random

from enum import Enum
from datetime import datetime as Datetime

class TrainTypes(Enum):
    TRAIN_REAL = 1
    TRAIN_FAKE = 2
    TEST_REAL = 3
    TEST_FAKE = 4

class Cache(object):
    def __init__(self, manager):
        self.manager = None
        self.cache = None

        self.train_fake_mels = None
        self.test_fake_mels = None

        if manager is not None:
            self.start(manager)

    def start(self, manager):
        self.manager = manager

        self.cache = manager.dict()
        self.cache[TrainTypes.TEST_REAL] = manager.dict()
        self.cache[TrainTypes.TEST_FAKE] = manager.dict()
        self.cache[TrainTypes.TRAIN_REAL] = manager.dict()
        self.cache[TrainTypes.TRAIN_FAKE] = manager.dict()

        self.train_fake_mels = manager.list()
        self.test_fake_mels = manager.list()

    def pop_samples(
        self, label, num_samples, is_training: bool
    ):
        assert label in (0, 1)

        if label == 0:
            if is_training:
                samples = self.cache[TrainTypes.TRAIN_REAL]
            else:
                samples = self.cache[TrainTypes.TEST_REAL]
        else:
            if is_training:
                samples = self.cache[TrainTypes.TRAIN_FAKE]
            else:
                samples = self.cache[TrainTypes.TEST_FAKE]

        all_keys = list(samples.keys())
        keys = random.sample(all_keys, k=num_samples)
        sample_images, sample_mels = [], []

        for key in keys:
            datapoint = samples[key]
            image, mel = datapoint
            sample_images.append(image)
            sample_mels.append(mel)
            del samples[key]

        return sample_images, sample_mels

    def add(self, train_type, key, image, mel):
        assert train_type in self.cache
        sub_cache = self.cache[train_type]
        sub_cache[key] = (image, mel)

    def add_train_fake_mel(self, mel):
        self.train_fake_mels.append(mel)

    def pop_fake_mel(self, is_training: bool):
        if is_training:
            return self.train_fake_mels.pop()
        else:
            return self.test_fake_mels.pop()

    @staticmethod
    def make_date_stamp():
        return Datetime.now().strftime("%y%m%d-%H%M")

    def safe_pop_fake_mel(self, is_training: bool):
        while True:
            try:
                return self.pop_fake_mel(is_training=is_training)
            except IndexError:
                stamp = self.make_date_stamp()
                print(f'POP mel FAILED {is_training} [{stamp}]')
                time.sleep(1)

    @property
    def train_fake_mel_size(self):
        return len(self.train_fake_mels)

    @property
    def test_fake_mel_size(self):
        return len(self.test_fake_mels)

    @property
    def min_mel_size(self):
        return min(
            self.train_fake_mel_size,
            self.test_fake_mel_size
        )

    def add_test_fake_mel(self, mel):
        self.test_fake_mels.append(mel)

    def get_size(self, train_type):
        return len(self.cache[train_type])

    @property
    def min_samples(self):
        return min(
            len(self.cache[TrainTypes.TRAIN_REAL]),
            len(self.cache[TrainTypes.TEST_REAL]),
            len(self.cache[TrainTypes.TRAIN_FAKE]),
            len(self.cache[TrainTypes.TEST_FAKE])
        )
