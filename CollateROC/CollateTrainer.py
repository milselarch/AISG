try:
    import rocstar
    from CollateModel import CollateModel
except ModuleNotFoundError:
    from . import rocstar
    from .CollateModel import CollateModel

try:
    # need it for tensorboard
    import tensorflow as tf
except ImportError:
    print('WARNING: NO TENSORFLOW FOUND')

import numpy as np

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import sklearn
import pandas as pd
import functools
import argparse
import math
import time
import cv2
import os

from tqdm.auto import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms
from datetime import datetime as Datetime
from sklearn import metrics

torch.cuda.empty_cache()

def round_sig(x, sig=2):
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


class CollateTrainer(object):
    def __init__(
        self, seed=420, test_p=0.05, use_cuda=True,
        valid_p=0.2, save_threshold=0.005,
        preload_path=None
    ):
        self.date_stamp = self.make_date_stamp()

        self.name = 'collator'
        self.batch_size = 128
        self.epochs = 50

        self.accum_train_score = 0
        self.accum_validate_score = 0
        self.save_threshold = save_threshold
        self.valid_p = valid_p
        self.test_p = test_p
        self.seed = seed

        self.save_best_every = 2500
        self.perf_decay = 0.96

        self.features = ['face_pred', 'audio_pred']
        self.tensorboard_started = False
        self.tfile_writer = None
        self.vfile_writer = None

        self.bce_episodes = 1000
        self.episode_no = 0

        torch.backends.cudnn.benchmark = True
        self.use_cuda = use_cuda

        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = CollateModel()
        self.model = self.model.to(self.device)
        # self.criterion = nn.CrossEntropyLoss()
        self.bce_criterion = nn.BCELoss()
        self.roc_criterion = rocstar.roc_star_loss

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.001,
            betas=(0.9, 0.999), eps=1e-08
        )
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )

        self.test_df = None
        self.train_df = None
        self.epoch_gamma = 0.20
        self.whole_y_pred = None
        self.whole_y_t = None
        self.last_whole_y_t = None
        self.last_whole_y_pred = None

        self.load_dataset()
        if preload_path is not None:
            self.load_model(preload_path)

    def load_dataset(self):
        face_path = '../stats/vid-face-preds-211106-0909.csv'
        aud_path = '../FakeVoiceTorch/csvs/aisg-preds-211013-1810.csv'

        face_df = pd.read_csv(face_path)
        face_df['face_pred'] = face_df['median']
        aud_df = pd.read_csv(aud_path)
        aud_df['audio_pred'] = aud_df['median_pred']

        features = [
            'filename', 'face_pred', 'label', 'audio_fake',
            'face_fake', 'swap_fake'
        ]

        combine_df = pd.merge(
            left=aud_df[['filename', 'audio_pred']],
            right=face_df[features],
            left_on='filename', right_on='filename'
        )

        sub_df = combine_df[combine_df['swap_fake'] == 0]
        sub_df = sub_df[['face_pred', 'audio_pred', 'label']]
        train_df, test_df, _, _ = train_test_split(
            sub_df, sub_df, random_state=self.seed
        )

        self.test_df = test_df
        self.train_df = train_df

    def prepare_batch(
        self, batch_size, is_training=True, fake_p=None,
        randomize=True
    ):
        if is_training:
            df = self.train_df
        else:
            df = self.test_df

        while True:
            sub_df = df.sample(batch_size)
            np_inputs = sub_df[self.features].to_numpy()
            np_labels = sub_df['label'].to_numpy()
            if len(np.unique(np_labels)) == 2:
                break

        np_inputs = np_inputs.astype(np.float32)
        np_labels = np_labels.flatten()
        return np_inputs, np_labels

    def load_model(self, model_path, eval_mode=True):
        self.model.load_state_dict(torch.load(
            model_path, map_location=self.device
        ))
        if eval_mode:
            self.model.eval()

    @property
    def criterion(self):
        if self.use_roc:
            return functools.partial(
                self.roc_criterion,
                gamma=self.epoch_gamma,
                _epoch_true=self.last_whole_y_t,
                epoch_pred=self.last_whole_y_t
            )
        else:
            return self.bce_criterion

    def batch_train(
        self, episode_no, batch_size=None, fake_p=0.5,
        record=False
    ):
        if batch_size is None:
            batch_size = self.batch_size

        self.model.train()
        batch_x, np_labels = self.prepare_batch(
            batch_size=batch_size, fake_p=fake_p,
            is_training=True, randomize=True
        )

        torch_batch_x = torch.tensor(batch_x)
        torch_batch_x = torch_batch_x.to(self.device).float()
        torch_labels = torch.tensor(np_labels).float()
        torch_labels = torch_labels.to(self.device).detach()
        self.optimizer.zero_grad()

        preds = self.model(torch_batch_x)
        loss = self.criterion(preds, torch_labels)
        loss.backward()
        loss_value = loss.item()

        params = self.model.parameters()
        torch.nn.utils.clip_grad_norm_(params, 0.5)
        self.optimizer.step()

        if self.use_roc:
            self.epoch_y_pred.extend(preds)
            self.epoch_y_t.extend(y)

        callback = self.record_train_errors if record else None
        np_preds = preds.detach().cpu().numpy().flatten()
        flat_labels = np_labels.flatten()

        self.whole_y_pred = np.append(self.whole_y_pred, np_preds)
        self.whole_y_t = np.append(self.whole_y_t, flat_labels)

        score = self.record_metrics(
            episode_no, loss_value, np_preds, flat_labels,
            callback=callback
        )

        return score

    def batch_validate(
        self, episode_no, batch_size=None, fake_p=0.5
    ):
        if batch_size is None:
            batch_size = self.batch_size

        batch_x, np_labels = self.prepare_batch(
            batch_size=batch_size, fake_p=fake_p,
            is_training=False, randomize=True
        )

        torch_batch_x = torch.tensor(batch_x)
        torch_batch_x = torch_batch_x.to(self.device).float()
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

    def predict_file(self, filepath:str):
        assert type(filepath) is str
        predictions = self.batch_predict([filepath])
        prediction = predictions[0][0]
        return prediction

    def predict_images(self, image_list, to_numpy=True):
        images_preds = []

        for input_arr in image_list:
            assert type(input_arr) is torch.Tensor
            input_arr = input_arr.to(self.device)
            if len(input_arr.shape) == 3:
                input_arr = torch.unsqueeze(input_arr, 0)

            preds = self.batch_predict(input_arr)
            preds = preds.flatten()
            assert len(preds) == 1
            images_preds.append(preds[0])

        if to_numpy:
            images_preds = np.array(images_preds)

        return images_preds

    def predict(self, data, no_grad=False):
        if not no_grad:
            return self.model(data)

        with torch.no_grad():
            return self.model(data)

    @property
    def use_roc(self):
        return (
            (self.episode_no > self.bce_episodes) and
            (self.last_whole_y_pred is not None) and
            (self.last_whole_y_t is not None)
        )

    def train(
        self, episodes=10 * 1000, batch_size=None,
        fake_p=0.5
    ):
        if batch_size is None:
            batch_size = self.batch_size

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

        _, torch_labels = self.prepare_batch(
            batch_size=batch_size, is_training=True
        )

        self.whole_y_pred = np.array([])
        self.whole_y_t = np.array([])
        self.epoch_gamma = 0.20
        epoch = -1

        while episode_no <= episodes:
            self.episode_no = episode_no
            tag = 'R' if self.use_roc else 'B'

            if run_validation:
                desc = f'[{tag}] VA episode {episode_no}/{episodes}'
                validate_eps += batch_size
                run_validation = False

                score = self.batch_validate(
                    episode_no, batch_size=batch_size, fake_p=fake_p
                )

                self.accum_validate(score, self.perf_decay)
            else:
                desc = f'[{tag}] TR episode {episode_no}/{episodes}'
                train_eps += batch_size

                validate_threshold = train_eps * self.valid_p
                run_validation = validate_threshold > validate_eps

                score = self.batch_train(
                    episode_no, batch_size=batch_size, fake_p=fake_p,
                    record=run_validation
                )

                if run_validation:
                    self.accum_train(score, self.perf_decay)
                    if self.use_roc:
                        self.last_whole_y_t = torch.tensor(
                            self.whole_y_t
                        ).cuda()
                        self.last_whole_y_pred = torch.tensor(
                            self.whole_y_pred
                        ).cuda()

                        self.epoch_gamma = rocstar.epoch_update_gamma(
                            self.last_whole_y_t,
                            self.last_whole_y_pred, epoch
                        )

                        self.whole_y_pred = np.array([])
                        self.whole_y_t = np.array([])

                    epoch += 1

            pbar.set_description(desc)
            pbar.update(batch_size)
            episode_no += batch_size
            # time.sleep(0.1)

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
        round_train = round_sig(train_score, sig=3)
        round_validate = round_sig(validate_score, sig=3)

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

    @staticmethod
    def make_date_stamp():
        return Datetime.now().strftime("%y%m%d-%H%M")

    def tensorboard_start(self):
        log_dir = f'saves/logs/MES-{self.date_stamp}'
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

        auc = sklearn.metrics.roc_auc_score(flat_labels, np_preds)

        errors = abs(np_preds - flat_labels)
        me = np.sum(errors) / errors.size
        squared_error = np.sum((np_preds - flat_labels) ** 2)
        mse = (squared_error / errors.size) ** 0.5

        if callback is not None:
            callback(
                step=episode_no, loss=loss_value,
                me=me, mse=mse, accuracy=accuracy, auc=auc
            )

        score = 2 * auc - 1
        return score

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
        accuracy=-1, auc=-1
    ):
        assert self.tensorboard_started
        score = 2 * accuracy - 1

        with file_writer.as_default():
            tf.summary.scalar('loss', data=loss, step=step)
            tf.summary.scalar('mean_error', data=me, step=step)
            tf.summary.scalar('accuracy', data=accuracy, step=step)
            tf.summary.scalar('score-acc', data=score, step=step)
            tf.summary.scalar('AUC', data=auc, step=step)
            tf.summary.scalar(
                'mean_squared_error', data=mse, step=step
            )

            file_writer.flush()
