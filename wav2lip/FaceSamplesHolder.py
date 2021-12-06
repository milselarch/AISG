import random

try:
    import ParentImport
    import audio

except ModuleNotFoundError:
    from . import ParentImport
    from . import audio

import os
import sys
import pandas as pd
import numpy as np
import cProfile
import torch
import time
import gc

from tqdm.auto import tqdm


class Timer(object):
    def __init__(self, total_time=0):
        self.total_time = total_time
        self.start_time = None

    @property
    def total(self):
        return self.total_time

    def __repr__(self):
        name = self.__class__.__name__
        return f'{name}({self.total_time})'

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # warning: returning True in __exit__
        # WILL SUPPRESS ALL ERRORS
        self.pause()

    def start(self):
        self.start_time = time.perf_counter()

    def pause(self):
        assert self.start_time is not None
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        self.total_time += duration
        self.start_time = None


class FaceSamplesHolder(object):
    def __init__(
        self, predictor, batch_size=16, use_mouth_image=False,
        timer=None
    ):
        if timer is None:
            timer = Timer()

        self.predictor = predictor
        self.batch_size = batch_size
        self.use_mouth_image = use_mouth_image
        self.timer = timer

        self.face_samples_map = {}
        self.face_samples_cache = {}
        self.face_preds_map = {}
        self.face_confs_map = {}
        self.fps_cache = {}
        self.mel_cache = {}

    def make_video_preds(self, check=True):
        assert not check or (len(self.face_samples_map) == 0)
        video_preds_map = {}

        for key in self.face_preds_map:
            video_face_preds = self.face_preds_map[key]
            filename, face_no = key

            if filename not in video_preds_map:
                video_preds_map[filename] = {}

            video_preds = video_preds_map[filename]
            assert face_no not in video_preds
            video_preds[face_no] = video_face_preds

        return video_preds_map

    def add_face_sample(
        self, filename, face_samples, mel, face_no, fps
    ):
        key = (filename, face_no)
        self.mel_cache[filename] = mel
        self.face_samples_map[key] = face_samples
        self.fps_cache[filename] = fps
        has_predictions = True

        while has_predictions:
            has_predictions = self.auto_predict_samples(
                flush=False
            )

    def add_to_cache(self, key, face_sample):
        if len(self.face_samples_cache) < self.batch_size:
            if key not in self.face_samples_cache:
                self.face_samples_cache[key] = []

            cache_face_samples = self.face_samples_cache[key]
            cache_face_samples.append(face_sample)
            return True

        return False

    @staticmethod
    def warn(message):
        print(message, file=sys.stderr)

    @property
    def samples_left(self):
        samples_left = 0

        for key in self.face_samples_map:
            samples = self.face_samples_map[key]
            samples_left += len(samples)

        return samples_left

    def load_from_cache(self, num_samples, check_size=True):
        assert (
            num_samples <= len(self.face_samples_cache)
            or not check_size
        )

        img_batch, mel_batch = [], []
        init_cache_keys = list(self.face_samples_cache.keys())
        random.shuffle(init_cache_keys)
        cache_keys = init_cache_keys[::]

        if len(cache_keys) < num_samples:
            cache_size = len(cache_keys)
            samples_left = self.samples_left
            msg = f'UNDERSIZED CACHE {cache_size} - {samples_left}'
            self.warn(msg)

        while len(cache_keys) < num_samples:
            cache_keys += init_cache_keys

        for key in cache_keys[:num_samples]:
            filename, face_no = key
            face_samples = self.face_samples_cache[key]
            face_sample = random.choice(face_samples)

            cct = self.mel_cache[filename]
            fps = self.fps_cache[filename]
            t_img, t_mel = self.predictor.to_torch_batch(
                [face_sample], cct, fps=fps, auto_double=False
            )

            img_batch.append(t_img)
            mel_batch.append(t_mel)

        assert len(img_batch) == num_samples
        return img_batch, mel_batch

    def resolve_samples(
        self, check_size=True, use_mouth_image=False
    ):
        length = len(self.face_samples_map)
        if check_size and (length < self.batch_size):
            return False

        img_batch, mel_batch, key_batch = [], [], []
        keys = list(self.face_samples_map.keys())

        for key in keys:
            filename, face_no = key
            face_samples = self.face_samples_map[key]
            face_sample = face_samples.pop()

            cct = self.mel_cache[filename]
            fps = self.fps_cache[filename]
            torch_data = self.predictor.to_torch_batch(
                [face_sample], cct, fps=fps, auto_double=False,
                use_mouth_image=use_mouth_image
            )

            if len(self.face_samples_map[key]) == 0:
                del self.face_samples_map[key]

            if torch_data is None:
                continue

            t_img, t_mel = torch_data
            self.add_to_cache(key, face_sample)
            if key not in self.face_samples_map:
                if key not in self.face_samples_cache:
                    del self.mel_cache[filename]

            key_batch.append(key)
            img_batch.append(t_img)
            mel_batch.append(t_mel)

        buffer_needed = self.batch_size - len(img_batch)
        cache_items = self.load_from_cache(
            buffer_needed, check_size=check_size
        )

        cache_img_batch, cache_mel_batch = cache_items
        img_batch.extend(cache_img_batch)
        mel_batch.extend(cache_mel_batch)

        t_img_batch = torch.cat(img_batch)
        t_mel_batch = torch.cat(mel_batch)
        return t_img_batch, t_mel_batch, key_batch

    def auto_predict_samples(self, flush=False):
        check_size = not flush
        torch_batch = self.resolve_samples(
            check_size, use_mouth_image=self.use_mouth_image
        )

        if torch_batch is False:
            return False

        self.timer.start()
        t_img_batch, t_mel_batch, key_batch = torch_batch
        preds, confs = self.predictor.predict(t_mel_batch, t_img_batch)
        self.timer.pause()

        preds = preds.detach().cpu().numpy()
        confs = confs.detach().cpu().numpy()

        for k, key in enumerate(key_batch):
            prediction = preds[k]
            confidence = confs[k]

            if key not in self.face_preds_map:
                self.face_preds_map[key] = []
                self.face_confs_map[key] = []

            face_preds = self.face_preds_map[key]
            face_confs = self.face_confs_map[key]
            face_preds.append(prediction)
            face_confs.append(confidence)

        return True

    def flush(self):
        while len(self.face_samples_map) > 0:
            self.auto_predict_samples(flush=True)
