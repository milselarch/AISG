try:
    import ParentImport

    from Dataset import Dataset
    from network.classifier import *
    from network.transform import mesonet_data_transforms

except ModuleNotFoundError:
    from . import ParentImport

    from .Dataset import Dataset
    from .network.classifier import *
    from .network.transform import mesonet_data_transforms

import numpy as np

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import argparse
import time
import cv2
import os

from tqdm.auto import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from datetime import datetime as Datetime

torch.cuda.empty_cache()

def round_sig(x, sig=2):
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


class MesoTrainer(object):
    def __init__(
        self, seed=420, test_p=0.05, use_cuda=True,
        valid_p=0.05, load_dataset=True, save_threshold=0.01,
        use_inception=False, preload_path=None
    ):
        self.date_stamp = self.make_date_stamp()
        self.use_inception = use_inception

        self.name = 'Mesonet'
        self.batch_size = 64
        self.epochs = 50

        self.accum_train_score = 0
        self.accum_validate_score = 0
        self.save_threshold = save_threshold
        self.valid_p = valid_p
        self.test_p = test_p

        self.save_best_every = 2500
        self.perf_decay = 0.96

        self.tensorboard_started = False
        self.tfile_writer = None
        self.vfile_writer = None

        torch.backends.cudnn.benchmark = True
        self.use_cuda = use_cuda

        if use_inception:
            self.model = MesoInception4(
                num_classes=1, use_sigmoid=True
            )
        else:
            self.model = Meso4(
                num_classes=1, use_sigmoid=True
            )

        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.cuda()
        else:
            self.device = torch.device("cpu")

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.001,
            betas=(0.9, 0.999), eps=1e-08
        )
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )

        self.dataset = Dataset(seed=seed, load=load_dataset)
        if preload_path is not None:
            self.load_model(preload_path)

    def transform(self, image):
        return self.dataset.transform(image)

    def load_model(self, model_path, eval_mode=True):
        self.model.load_state_dict(torch.load(model_path))
        if eval_mode:
            self.model.eval()

    def batch_train(
        self, episode_no, batch_size=None, fake_p=0.5,
        record=False
    ):
        if batch_size is None:
            batch_size = self.batch_size

        self.model.train()
        batch_x, np_labels = self.dataset.prepare_batch(
            batch_size=batch_size, fake_p=fake_p,
            is_training=True, randomize=True
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
        self, episode_no, batch_size=None, fake_p=0.5
    ):
        if batch_size is None:
            batch_size = self.batch_size

        batch_x, np_labels = self.dataset.prepare_batch(
            batch_size=batch_size, fake_p=fake_p,
            is_training=False, randomize=True
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

    def batch_predict(
        self, batch_x, to_numpy=True
    ):
        if type(batch_x) is str:
            batch_x = [batch_x]

        if type(batch_x) in (list, tuple):
            assert type(batch_x[0]) is str
            labels = [1] * len(batch_x)
            batch_x, np_labels = self.dataset.load_batch(
                batch_x, labels
            )

        if type(batch_x) is np.ndarray:
            torch_batch_x = torch.tensor(batch_x)
        else:
            assert type(batch_x) is torch.Tensor
            torch_batch_x = batch_x

        self.model.eval()
        torch_batch_x = torch_batch_x.to(self.device)
        preds = self.model(torch_batch_x)
        if to_numpy:
            preds = preds.detach().cpu().numpy()

        return preds

    def load_batch(
        self, batch_filepaths, batch_labels
    ):
        batch_x, np_labels = self.dataset.load_batch(
            batch_filepaths, batch_labels
        )

        assert type(batch_x) is np.ndarray
        assert type(np_labels) is np.ndarray
        return batch_x, np_labels

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

        while episode_no <= episodes:
            if run_validation:
                desc = f'VA episode {episode_no}/{episodes}'
                validate_eps += batch_size
                run_validation = False

                score = self.batch_validate(
                    episode_no, batch_size=batch_size, fake_p=fake_p
                )

                self.accum_validate(score, self.perf_decay)
            else:
                desc = f'TR episode {episode_no}/{episodes}'
                train_eps += batch_size

                validate_threshold = train_eps * self.valid_p
                run_validation = validate_threshold > validate_eps

                score = self.batch_train(
                    episode_no, batch_size=batch_size, fake_p=fake_p,
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
