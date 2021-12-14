try:
    import ParentImport
    import audio

    from SyncDataset import SyncDataset
    from models import SyncNet_color as SyncNet
    from models import SyncnetJoonV1
    from models import SyncnetJoon
    from helpers import WeightedLogitsBCE
    from hparams import hparams

    from DeepfakeDetection.FaceExtractor import FaceImage

except ModuleNotFoundError:
    from . import ParentImport
    from . import audio

    from .SyncDataset import SyncDataset
    from .models import SyncNet_color as SyncNet
    from .models import SyncnetJoon
    from .models import SyncnetJoonV1
    from .helpers import WeightedLogitsBCE
    from .hparams import hparams

    from ..DeepfakeDetection.FaceExtractor import FaceImage

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import cProfile
import random
import torch.optim as optim
import python_speech_features
import argparse
import time
import math
import cv2
import os

print('IMPORT', time.time())

from tqdm.auto import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from datetime import datetime as Datetime
from torch.multiprocessing import set_start_method
from typing import Optional, List

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    print('TENSOR BOARD NOT FOUND')

# torch.cuda.empty_cache()

def round_sig(x, sig=2):
    return str(round(x, sig))


try:
    set_start_method('spawn')
except RuntimeError:
    pass


class SyncnetTrainer(object):
    def __init__(
        self, seed=420, test_p=0.05, use_cuda=True,
        valid_p=0.05, load_dataset=False, save_threshold=0.01,
        preload_path=None, is_checkpoint=True, syncnet_T=5,
        strict=True, train_workers=0, test_workers=0,
        pred_ratio=0., use_joon=False, mel_step_size=None,

        face_base_dir='../datasets/extract/mtcnn-sync',
        video_base_dir='../datasets/train/videos',
        audio_base_dir='../datasets/extract/audios-flac',
        mel_cache_path='saves/preprocessed/mel_cache.npy',

        old_joon=True, dropout_p=0.0, transform_image=False,
        fcc_list=None, predict_confidence=False,
        binomial_train_sampling=True, binomial_test_sampling=False,
        eval_mode=False
    ):
        self.date_stamp = self.make_date_stamp()

        self.name = 'color_syncnet'
        self.pred_ratio = pred_ratio
        self.syncnet_T = syncnet_T
        self.batch_size = 32
        self.epochs = 50

        self.accum_train_score = 0
        self.accum_validate_score = 0
        self.save_threshold = save_threshold
        self.predict_confidence = predict_confidence
        self.transform_image = transform_image
        self.dropout_p = dropout_p
        self.valid_p = valid_p
        self.test_p = test_p

        self.binomial_train_sampling = binomial_train_sampling
        self.binomial_test_sampling = binomial_test_sampling

        self.save_best_every = 2500
        self.perf_decay = 0.96

        self.tensorboard_started = False
        self.tfile_writer = None
        self.vfile_writer = None

        self.face_base_dir = face_base_dir
        self.video_base_dir = video_base_dir
        self.audio_base_dir = audio_base_dir
        self.mel_cache_path = mel_cache_path

        torch.backends.cudnn.benchmark = True
        self.use_joon = use_joon
        self.old_joon = old_joon
        self.use_cuda = use_cuda

        if use_joon:
            if old_joon:
                print('USING JOON V1')
                self.model = SyncnetJoonV1()
            else:
                print('USING JOON V2')
                outputs = 2 if predict_confidence else 1
                self.model = SyncnetJoon(
                    dropout_p=dropout_p, fcc_list=fcc_list,
                    num_outputs=outputs
                )

            if mel_step_size is None:
                mel_step_size = 20
        else:
            print('USING OLD SYNCNET')
            self.model = SyncNet(self.syncnet_T)
            if mel_step_size is None:
                mel_step_size = 16

        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.cuda()
        else:
            self.device = torch.device("cpu")

        grad_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]

        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCELoss()
        self.criterion = WeightedLogitsBCE()
        self.params = list(self.model.parameters())

        self.optimizer = optim.Adam(
            self.params, lr=hparams.initial_learning_rate,
            betas=(0.9, 0.999), eps=1e-08
        )
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )

        self.load_dataset = load_dataset
        self.mel_step_size = mel_step_size
        self.dataset = SyncDataset(
            seed=seed, load=load_dataset,
            train_workers=train_workers, test_workers=test_workers,
            mel_step_size=mel_step_size, use_joon=self.use_joon,
            face_base_dir=self.face_base_dir,
            video_base_dir=self.video_base_dir,
            audio_base_dir=self.audio_base_dir,
            mel_cache_path=self.mel_cache_path,
            transform_image=self.transform_image
        )
        
        if preload_path is not None:
            if is_checkpoint:
                self.load_checkpoint(preload_path, strict=strict)
            else:
                self.load_model(preload_path)

        if eval_mode:
            self.enter_val_model()

    def make_data_loader(self, *args, **kwargs):
        return self.dataset.make_data_loader(*args, **kwargs)

    def start_dataset_workers(self, *args, **kwargs):
        if not self.dataset.loaded:
            self.dataset.load(*args, **kwargs)

    def transform(self, image):
        return self.dataset.transform(image)

    def load_model(self, model_path, eval_mode=False):
        self.model.load_state_dict(torch.load(
            model_path, map_location=self.device
        ))
        if eval_mode:
            self.enter_val_model()

        print(f'PRELOADED SYNCNET FROM {model_path}')

    def enter_val_model(self):
        self.model.eval()

    def load_parameters(self, model_path):
        state = self.model.state_dict()
        loaded_state = torch.load(model_path)

        for name, param in loaded_state.items():
            orig_name = name

            if name not in state:
                name = name.replace("module.", "")

                if name not in state:
                    print("%s is not in the model." % orig_name)
                    continue

            if state[name].size() != loaded_state[orig_name].size():
                print(
                    f'Wrong parameter length: {orig_name}'
                    f'model: {self_state[name].size()},'
                    f'loaded: {loaded_state[orig_name].size()}'
                )
                continue

            state[name].copy_(param)

    def cosine_loss(self, audio_embed, face_embed, labels):
        d = nn.functional.cosine_similarity(audio_embed, face_embed)
        loss = self.criterion(d.unsqueeze(1), labels)
        return loss

    def predict(self, t_mels, t_images, sigmoid=True):
        if self.pred_ratio == 0.:
            # assert not self.use_joon
            raw_preds = self.model.predict(t_mels, t_images)
        else:
            raw_preds = self.model.pred_fcc(
                t_mels, t_images, fcc_ratio=self.pred_ratio,
                sigmoid=sigmoid
            )

        if self.predict_confidence:
            predictions = raw_preds[:, 0]
            confidences = raw_preds[:, 1]
        else:
            non_flat_dims = 0
            for dim in raw_preds.shape:
                non_flat_dims += 1 if dim != 1 else 0

            assert non_flat_dims == 1
            predictions = torch.flatten(raw_preds)
            confidences = torch.ones(predictions.shape)

        return predictions, confidences

    def batch_train(
        self, episode_no, batch_size=None, fake_p=0.5,
        record=False
    ):
        if batch_size is None:
            batch_size = self.batch_size

        self.model.train()
        self.optimizer.zero_grad()

        torch_batch = self.prepare_torch_batch(
            batch_size=batch_size, fake_p=fake_p,
            binomial_sampling=self.binomial_train_sampling,
            is_training=True, randomize=True
        )

        t_labels, t_images, t_mels = torch_batch
        raw_output = self.predict(t_mels, t_images, sigmoid=False)
        sigmoid_preds = [torch.sigmoid(x) for x in raw_output]
        preds, confs = sigmoid_preds

        raw_preds, raw_confs = raw_output
        loss = self.criterion(raw_preds, t_labels, confs)
        loss_value = loss.item()

        loss.backward()
        self.optimizer.step()

        np_confs = confs.detach().cpu().numpy().flatten()
        np_preds = preds.detach().cpu().numpy().flatten()
        flat_labels = t_labels.detach().cpu().numpy().flatten()

        callback = self.record_train_errors if record else None
        score = self.record_metrics(
            episode_no, loss_value, np_preds, flat_labels,
            callback=callback, confs=np_confs
        )

        return score

    def batch_validate(
        self, episode_no, batch_size=None, fake_p=0.5,
        enter_eval_mode=True, use_train_data=False
    ):
        if batch_size is None:
            batch_size = self.batch_size

        # self.optimizer.zero_grad()
        if enter_eval_mode:
            self.model.train(False)
        else:
            self.model.train(True)

        torch_batch = self.prepare_torch_batch(
            batch_size=batch_size, fake_p=fake_p,
            binomial_sampling=self.binomial_test_sampling,
            is_training=use_train_data, randomize=True
        )

        with torch.no_grad():
            t_labels, t_images, t_mels = torch_batch
            raw_output = self.predict(t_mels, t_images, sigmoid=False)
            sigmoid_preds = [torch.sigmoid(x) for x in raw_output]
            preds, confs = sigmoid_preds

            raw_preds, raw_confs = raw_output
            loss = self.criterion(raw_preds, t_labels, confs)
            loss_value = loss.item()

        self.model.train(True)
        self.optimizer.zero_grad()

        np_confs = confs.detach().cpu().numpy().flatten()
        np_preds = preds.detach().cpu().numpy().flatten()
        flat_labels = t_labels.detach().cpu().numpy().flatten()

        callback = self.record_validate_errors
        score = self.record_metrics(
            episode_no, loss_value, np_preds, flat_labels,
            callback=callback, confs=np_confs
        )

        return score

    def prepare_batch(self, *args, **kwargs):
        return self.dataset.prepare_batch(*args, **kwargs)

    def prepare_torch_batch(self, *args, **kwargs):
        batch = self.prepare_batch(*args, **kwargs)
        torch_batch = self.dataset.torchify_batch(*batch)
        t_labels, t_images, t_mels = torch_batch

        t_mels = t_mels.to(self.device)
        t_images = t_images.to(self.device)
        t_labels = t_labels.to(self.device).detach()

        return t_labels, t_images, t_mels

    def load_checkpoint(
        self, checkpoint_path, reset_optimizer=False,
        strict=True
    ):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["state_dict"]

        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v

        self.model.load_state_dict(new_state_dict, strict=strict)

        if reset_optimizer:
            optimizer_state = checkpoint["optimizer"]
            if optimizer_state is not None:
                print(f"Loaded optimizer state")
                self.optimizer.load_state_dict(
                    checkpoint["optimizer"]
                )

        print(f'Loaded checkpoint from {checkpoint_path}')

    def to_torch_batch(
        self, face_samples: List[List[FaceImage]],
        melspectogram, fps, transpose_audio=False, use_joon=None,
        is_raw_audio=False, to_device=True, auto_double=True,
        use_mouth_image=False, swap_rgb=False
    ):
        if use_joon is None:
            use_joon = self.use_joon

        if is_raw_audio:
            if use_joon:
                melspectogram = self.load_cct(melspectogram)
            else:
                melspectogram = audio.melspectrogram(
                    melspectogram
                ).T

        img_batch, mel_batch = [], []

        for sample_batch in face_samples:
            first_face_image = sample_batch[0]
            frame_no = first_face_image.frame_no
            raw_images = []

            for face_image in sample_batch:
                if use_mouth_image:
                    raw_image = face_image.mouth_image
                else:
                    raw_image = face_image.image

                assert raw_image is not None
                raw_images.append(raw_image)

            if use_joon:
                torch_img_sample = self.dataset.batch_images_joon(
                    raw_images, mirror_prob=0, swap_rgb=swap_rgb
                )
                torch_mel_sample = self.dataset.load_mel_batch_joon(
                    melspectogram, fps=fps, frame_no=frame_no
                )
            else:
                torch_img_sample = self.dataset.batch_images_wav2lip(
                    raw_images, mirror_prob=0
                )
                torch_mel_sample = self.dataset.load_mel_batch_wav2lip(
                    melspectogram, fps=fps, frame_no=frame_no,
                    transpose=transpose_audio
                )

            # print(torch_mel_sample.shape[-1])
            if self.dataset.is_incomplete_mel(torch_mel_sample):
                continue

            img_batch.append(torch_img_sample)
            mel_batch.append(torch_mel_sample)
            # print('TI', torch_img_sample.shape)
            # print('TM', torch_mel_sample.shape)

        if auto_double and (len(img_batch) == 1):
            print('SINGLE SAMPLE ONLY')
            img_batch = [img_batch[0], img_batch[0]]
            mel_batch = [mel_batch[0], mel_batch[0]]

        if len(img_batch) == 0:
            return None

        t_img_batch = torch.cat(img_batch)
        t_mel_batch = torch.cat(mel_batch)

        if to_device:
            t_img_batch = t_img_batch.to(self.device)
            t_mel_batch = t_mel_batch.to(self.device)

        return t_img_batch, t_mel_batch

    def face_predict(
        self, *args, to_numpy=False, predict_distance=False,
        no_grad=True, **kwargs
    ):
        t_img_batch, t_mel_batch = self.to_torch_batch(
            *args, **kwargs
        )

        if predict_distance:
            is_joon_model = isinstance(self.model, SyncnetJoon)
            is_old_joon_model = isinstance(self.model, SyncnetJoonV1)
            assert is_joon_model or is_old_joon_model
            predict = self.model.predict_distance
        else:
            predict = self.predict

        self.model.eval()

        if no_grad:
            with torch.no_grad():
                raw_preds = predict(t_mel_batch, t_img_batch)
        else:
            raw_preds = predict(t_mel_batch, t_img_batch)

        predictions, confidences = raw_preds
        predictions = predictions.detach()

        if to_numpy:
            predictions = predictions.cpu().numpy()

        return predictions

    def face_predict_joon(
        self, face_samples, cct, fps, to_numpy=False,
        is_raw_audio=False, predict_distance=False
    ):
        return self.face_predict(
            face_samples=face_samples, melspectogram=cct,
            to_numpy=to_numpy, is_raw_audio=is_raw_audio,
            predict_distance=predict_distance, fps=fps,
            use_joon=True
        )

    @staticmethod
    def load_cct(*args, **kwargs):
        return SyncDataset.load_cct(*args, **kwargs)

    def predict_file(self, filepath: str):
        raise NotImplemented

    def predict_images(self, image_list, to_numpy=True):
        raise NotImplemented

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

    def ptrain(
        self, *args, profile_dir='saves/profiles', **kwargs
    ):
        stamp = self.make_date_stamp()
        assert os.path.isdir(profile_dir)
        profile_path = f'{profile_dir}/sync-train-{stamp}.profile'
        profile = cProfile.Profile()
        profile.enable()

        try:
            self.train(*args, **kwargs)
        except Exception as e:
            print('TRAINING FAILED')
            raise e
        finally:
            profile.disable()
            profile.dump_stats(profile_path)
            print(f'profile saved to {profile_path}')

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

    def validate(
        self, episodes=10 * 1000, batch_size=None,
        fake_p=0.5, use_train_data=False, mono_filename=False
    ):
        if batch_size is None:
            batch_size = self.batch_size

        episode_no = 0
        self.model.train(False)
        pbar = tqdm(range(episodes))
        batch_preds, batch_labels = [], []

        while episode_no <= episodes:
            desc = f'VA episode {episode_no}/{episodes}'
            pbar.set_description(desc)
            pbar.update(batch_size)

            if mono_filename:
                if use_train_data:
                    filelist = self.dataset.train_face_files
                else:
                    filelist = self.dataset.test_face_files

                filename = random.choice(filelist)
            else:
                filename = None

            torch_batch = self.prepare_torch_batch(
                batch_size=batch_size, fake_p=fake_p,
                is_training=use_train_data, randomize=True,
                filename=filename
            )

            with torch.no_grad():
                t_labels, t_images, t_mels = torch_batch
                raw_preds = self.predict(t_mels, t_images)
                preds, confs = raw_preds

            np_preds = preds.detach().cpu().numpy().flatten()
            np_labels = t_labels.detach().cpu().numpy().flatten()
            batch_preds.append(np_preds)
            batch_labels.append(np_labels)

            episode_no += batch_size

        all_preds = np.concatenate(batch_preds)
        all_labels = np.concatenate(batch_labels)
        me, mse, accuracy = self.get_metrics(all_preds, all_labels)
        mean_pred = np.mean(all_preds)

        print(f'accuracy: {accuracy}')
        print(f'mean error: {me}')
        print(f'mean squared error: {mse}')
        print(f'average pred: {mean_pred}')

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
        round_train = float(round_sig(train_score, sig=2))
        round_validate = float(round_sig(validate_score, sig=2))

        if round_train == int(round_train):
            round_train = int(round_train)
        if round_validate == int(round_validate):
            round_validate = int(round_validate)
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
        log_dir = f'saves/logs/SYN-{self.date_stamp}'
        train_path = log_dir + '/training'
        valid_path = log_dir + '/validation'
        self.tfile_writer = SummaryWriter(train_path)
        self.vfile_writer = SummaryWriter(valid_path)
        self.tensorboard_started = True

    @staticmethod
    def get_metrics_v1(np_preds, np_labels):
        round_preds = np.round(np_preds)
        matches = np.equal(round_preds, np_labels)
        accuracy = sum(matches) / len(matches)

        errors = abs(np_preds - np_labels)
        me = np.sum(errors) / errors.size
        squared_error = np.sum((np_preds - np_labels) ** 2)
        mse = (squared_error / errors.size) ** 0.5
        return me, mse, accuracy

    @staticmethod
    def get_metrics(np_preds, np_labels, weights=None):
        assert np_preds.shape == np_labels.shape

        if weights is None:
            weights = np.ones_like(np_preds)
            weights = weights.astype(np.float)

        n_weights = weights * len(weights) / sum(weights)

        round_preds = np.round(np_preds)
        matches = np.equal(round_preds, np_labels)
        w_accuracy = np.sum(n_weights * matches) / len(matches)

        errors = abs(np_preds - np_labels)
        w_errors = n_weights * errors
        w_me = np.sum(w_errors) / errors.size

        squared_error = np.sum((np_preds - np_labels) ** 2)
        w_squared_error = np.sum(n_weights * squared_error)
        w_mse = (w_squared_error / errors.size) ** 0.5
        return w_me, w_mse, w_accuracy

    def record_metrics(
        self, episode_no, loss_value, np_preds, flat_labels,
        callback, confs=None
    ):
        conf_mean, conf_std = None, None
        w_me, w_mse, w_accuracy = None, None, None
        me, mse, accuracy = self.get_metrics(
            np_preds, flat_labels
        )

        if self.predict_confidence and (confs is not None):
            n_confs = confs * len(confs) / np.sum(confs)
            conf_mean, conf_std = np.mean(confs), np.std(n_confs)
            w_me, w_mse, w_accuracy = self.get_metrics(
                np_preds, flat_labels, weights=n_confs
            )

        if callback is not None:
            callback(
                step=episode_no, loss=loss_value,
                me=me, mse=mse, accuracy=accuracy,
                w_me=w_me, w_mse=w_mse, w_accuracy=w_accuracy,
                conf_mean=conf_mean, conf_std=conf_std
            )

        if w_accuracy is not None:
            score = 2 * w_accuracy - 1
        else:
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
        accuracy=-1, w_loss=None, w_me=None, w_mse=None,
        w_accuracy=None, conf_mean=None, conf_std=None
    ):
        assert self.tensorboard_started
        score = 2 * accuracy - 1
        if w_accuracy is not None:
            w_score = 2 * w_accuracy - 1
        else:
            w_score = None

        def write_scalar(name, value, global_step=step):
            if value is None:
                return False

            file_writer.add_scalar(
                name, value, global_step=global_step
            )

        write_scalar('loss', loss)
        write_scalar('mean_error', me)
        write_scalar('mean_squared_error', mse)
        write_scalar('accuracy', accuracy)
        write_scalar('score', score)

        write_scalar('w_loss', w_loss)
        write_scalar('w_mean_error', w_me)
        write_scalar('w_mean_squared_error', w_mse)
        write_scalar('w_accuracy', w_accuracy)
        write_scalar('w_score', w_score)

        write_scalar('conf_mean', conf_mean)
        write_scalar('conf_std', conf_std)

        file_writer.flush()
