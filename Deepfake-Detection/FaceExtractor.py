import os

import numpy as np

import ParentImport
import datasets
import torchvision.transforms as transforms
import face_recognition
import pandas as pd
import cProfile
import datetime
import torch
import json
import dlib
import cv2

from PIL import Image
from tqdm.auto import tqdm
from matplotlib.pyplot import imshow
from network.models import model_selection
from detect_from_video import predict_with_model
# from trainer import Trainer


class Area(object):
    def __init__(self, row):
        assert row is not None
        self.row = row

    @property
    def x0(self):
        return self.row["left"]

    @property
    def x1(self):
        return self.row["right"]

    @property
    def y0(self):
        return self.row["top"]

    @property
    def y1(self):
        return self.row["bottom"]

    def __matmul__(self, other):
        return self.get_area(other)

    def get_area(self, other):
        x0 = max(self.x0, other.x0)
        x1 = min(self.x1, other.x1)
        y0 = max(self.y0, other.y0)
        y1 = min(self.y1, other.y1)

        if x1 - x0 < 0:
            return 0
        if y1 - y0 < 0:
            return 0

        return (y1 - y0) * (x1 - x0)

    def __repr__(self):
        return f'{self.__class__.__name__}{self.coords}'

    @property
    def coords(self):
        return self.x0, self.x1, self.y0, self.y1

    def centroid(self):
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def distance(self, other):
        x0, y0 = self.centroid()
        x1, y1 = other.centroid()
        squared_dist = (x1 - x0) ** 2 + (y1 - y0) ** 2
        return squared_dist ** 0.5


class FaceExtractor(object):
    def __init__(self, scale_down=1):
        dt = datetime.datetime.now()
        self.stamp = dt.strftime('%Y%m%d-%H%M%S')
        dataset = datasets.Dataset(basedir='../datasets')
        self.dataset = dataset
        # dataset.all_videos

        # print(dataframe.head())
        print(len(dataset.train_videos), dataset.train_videos[:5])
        print(len(dataset.test_videos), dataset.test_videos[:5])

        self.max_face_mapping = {}
        # self.num_videos = len(self.dataset.all_videos)
        # num_videos = 10

        self.scale_down = scale_down
        self.every_n_frames = 20
        self.n = self.every_n_frames
        self.cuda = True

        assert type(self.scale_down) is int
        assert self.scale_down >= 1
        self.rescale = 1 / self.scale_down

        self.output_path = f'../stats/detections-{self.stamp}.csv'
        self.profile_path = f'../stats/extract-{self.stamp}.profile'
        self.base_filename = "detections-20210903-230613.csv"
        self.base_faces = pd.read_csv(f'../stats/{self.base_filename}')
        self.base_faces['prediction'] = None
        self.invalid_videos = []

        print(f'rescale is {self.rescale}')
        print(f'CSV filename is {self.output_path}')

        self.model = model_selection(
            modelname='xception', num_out_classes=2, dropout=0.5
        )

        self.model_path = './pretrained_model/deepfake_c0_xception.pkl'
        self.model.load_state_dict(torch.load(self.model_path))
        # self.model = self.model.cuda() if cuda else self.model

    @staticmethod
    def extract_coords(index, face_rows, every_n_frames):
        min_top = float('inf')
        min_left = float('inf')
        max_right = float('-inf')
        max_bottom = float('-inf')

        for offset in (0,):
            i = index + offset

            if (i == -1) or (i == len(face_rows)):
                continue

            row = face_rows[i]

            top = int(row["top"])
            left = int(row["left"])
            right = int(row["right"])
            bottom = int(row["bottom"])

            min_top = min(min_top, top)
            min_left = min(min_left, left)
            max_right = max(max_right, right)
            max_bottom = max(max_bottom, bottom)

        return (
            min_top, min_left, max_right, max_bottom
        )

    @staticmethod
    def collate_faces(video_frame_rows, num_faces):
        face_frames = {}
        frames = video_frame_rows["frames"].to_numpy()
        frames = np.unique(frames)

        for frame_no in frames:
            # go through each frame time step
            frame_rows = video_frame_rows[
                video_frame_rows["frames"] == frame_no
            ]

            excluded_faces = []
            excluded_indexes = []

            for face_no in range(num_faces):
                if face_no not in face_frames:
                    face_frames[face_no] = []

                if face_no in excluded_faces:
                    continue

                # go through each face
                face_frame_list = face_frames[face_no]
                max_overlap_area = 0
                max_overlap_index = None
                max_overlap_row = None

                min_distance = float('inf')
                closest_index = None
                closest_row = None

                for index in frame_rows.index:
                    if index in excluded_indexes:
                        continue

                    # go through each frame image
                    row = frame_rows.loc[index]

                    if len(face_frame_list) == 0:
                        max_overlap_index = index
                        max_overlap_row = row
                        break

                    target_region = Area(face_frame_list[-1])
                    current_region = Area(row)
                    overlap_area = target_region.get_area(
                        current_region
                    )

                    if overlap_area > max_overlap_area:
                        max_overlap_area = overlap_area
                        max_overlap_index = index
                        max_overlap_row = row

                    distance = target_region.distance(current_region)
                    if distance < min_distance:
                        min_distance = distance
                        closest_index = index
                        closest_row = row

                if max_overlap_row is not None:
                    face_frame_list.append(max_overlap_row)
                    # excluded_faces.append(face_no)
                    excluded_indexes.append(max_overlap_index)
                elif closest_row is not None:
                    face_frame_list.append(closest_row)
                    # excluded_faces.append(face_no)
                    excluded_indexes.append(closest_index)

        return face_frames

    def export_face_frames(
        self, face_frames, num_faces, rescale_ratios,
        np_frames, base_dir
    ):
        for face_no in range(num_faces):
            #  print(f'FRAME NO {i} {np_frames.shape}')
            face_rows = face_frames[face_no]

            # print('rows', face_rows)

            for i, row in enumerate(face_rows):
                frame_no = row['frames']
                index = frame_no // self.every_n_frames
                frane = np_frames[index]
                # print('ROW', row)

                top, left, right, bottom = self.extract_coords(
                    i, face_rows, self.every_n_frames
                )

                if top == float('inf'):
                    continue

                area = (bottom - top) * (right - left)
                area_root = area ** 0.5
                buffer = int(area_root // 7)

                rescale = self.rescale
                b_top = int(rescale * max(top - buffer, 0))
                b_left = int(rescale * max(left - buffer, 0))
                b_right = int(rescale * (right + buffer))
                b_bottom = int(rescale * (bottom + buffer))

                face_crop = frane[b_top:b_bottom, b_left:b_right]

                if rescale_ratios is not None:
                    x_scale, y_scale = rescale_ratios
                    new_width = int(face_crop.shape[1] * x_scale)
                    new_height = int(face_crop.shape[0] * y_scale)
                    face_crop = cv2.resize(
                        face_crop, (new_width, new_height)
                    )

                im = Image.fromarray(face_crop)
                path = f'{base_dir}/{face_no}-{frame_no}.jpg'
                im.save(path)

    def extract_faces(
        self, filenames=None, export_dir='../datasets-local/faces',
        pre_resize=False
    ):
        if filenames is None:
            filenames = self.dataset.all_videos

        pbar = tqdm(range(len(filenames)))
        faceless_videos = []

        for k in pbar:
            filename = filenames[k]
            name = filename[:filename.index('.')]
            base_dir = f'{export_dir}/{name}'
            pbar.set_description(f"Processing {filename}")

            try:
                vid_obj = self.dataset.read_video(
                    filename, every_n_frames=self.every_n_frames,
                    scale=self.rescale
                )
            except datasets.FailedVideoRead:
                print(f'FILENAME LOAD FAILED {filename}')
                continue
            except ValueError as e:
                self.invalid_videos.append(filename)
                print(f'VALUE ERROR {filename} {k}')
                raise e

            if pre_resize:
                vid_obj = vid_obj.auto_resize()
                rescale_ratios = None
            else:
                rescale_ratios = vid_obj.get_rescale_ratios()

            np_frames = vid_obj.out_video
            video_frame_rows = self.base_faces[
                self.base_faces['filename'] == filename
            ]

            if len(video_frame_rows) == 0:
                print(f'FACELESS VIDEO: {filename}')
                faceless_videos.append(filename)
                continue

            if not os.path.exists(base_dir):
                os.mkdir(base_dir)

            # print('FRAME ROWS', video_frame_rows)
            # print('FRAMES', frames_column)
            # print(240 in frames_column)

            num_faces = video_frame_rows['num_faces'].to_numpy()[0]
            face_frames = self.collate_faces(
                video_frame_rows, num_faces
            )

            for face_no in tuple(face_frames.keys()):
                if len(face_frames) == 1:
                    break

                frames = face_frames[face_no]
                if len(frames) < 5:
                    del face_frames[face_no]

            left_positions = []
            for face_no in tuple(face_frames.keys()):
                left_positions.append(
                    face_frames[face_no][0]["left"]
                )

            left_positions = np.array(left_positions)
            sort_keys = np.argsort(left_positions)
            sorted_face_frames = {
                sort_keys[k]: face_frames[k]
                for k in range(num_faces)
            }

            self.export_face_frames(
                sorted_face_frames, num_faces,
                rescale_ratios, np_frames, base_dir
            )

        # base_faces.to_csv(output_path, index=False)
        # print(f'SAVED TO {output_path}')
        print(f'INVALID VIDEOS', self.invalid_videos)
        print(f'FACELESS VIDEOS', faceless_videos)
