try:
    from DeepfakeDetection.FaceExtractor import FaceExtractor
except ModuleNotFoundError:
    from .DeepfakeDetection.FaceExtractor import FaceExtractor

import loader
import datasets
import torch
import numpy as np
import pandas as pd
import time
import cv2
import os

from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from datetime import datetime
from tqdm.auto import tqdm

class NeuralFaceExtract(object):
    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=100,
            thresholds=[0.6, 0.7, 0.7], factor=0.709,
            post_process=True, device=self.device
        )
        self.df = None
        self.filename_log = None
        self.num_face_log = None
        self.frame_log = None
        self.face_log = None

        self.top_log = None
        self.left_log = None
        self.right_log = None
        self.bottom_log = None

    @staticmethod
    def make_date_stamp():
        return datetime.now().strftime("%y%m%d-%H%M")

    def fill_face_maps(self, np_frames, interval, batch_size):
        frame_face_boxes, index = [], 0
        face_confs = []

        while index < len(np_frames):
            end_index = index + batch_size
            batch = np_frames[index:end_index]
            bboxes, bconfs = self.mtcnn.detect(batch)
            face_confs.extend(bconfs)
            frame_face_boxes.extend(bboxes)
            index = end_index

        # assert len(frame_face_boxes) == len(np_frames)
        max_faces, face_mapping = 0, {}

        for k in range(len(frame_face_boxes)):
            frame_no = interval * k
            face_locations = []
            face_mapping[frame_no] = face_locations

            bboxes = frame_face_boxes[k]
            bconfs = face_confs[k]

            if bboxes is None:
                print(f'BBOXES IS NONE')
                break

            assert bconfs is not None
            assert bboxes is not None
            for bbox, bconf in zip(bboxes, bconfs):
                if bconf < 0.991:
                    # face detection confidence threshold
                    continue

                bbox = bbox.astype(int)
                bbox = np.clip(bbox, a_max=999999, a_min=0)
                # print(f'BBOX {k} {bbox} {bconf}')
                left, top, right, bottom = bbox

                assert right > left
                assert bottom > top
                face_locations.append((
                    top, right, bottom, left
                ))

            num_faces = len(face_locations)
            max_faces = max(max_faces, num_faces)

        return max_faces, face_mapping

    def process_filepaths(
        self, filepaths, callback=lambda *args: None,
        every_n_frames=20, batch_size=16, base_dir=None
    ):
        def fill_face_maps(frames, interval):
            return self.fill_face_maps(
                frames, interval, batch_size=batch_size
            )

        pbar = tqdm(filepaths)

        for filepath in pbar:
            if base_dir is not None:
                filepath = f'{base_dir}/{filepath}'

            video_cap = cv2.VideoCapture(filepath)
            width_in = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_in = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            scale = 0.5
            if min(width_in, height_in) < 700:
                scale = 1

            vid_obj = loader.load_video(
                filepath, every_n_frames=20, scale=scale
            )

            if vid_obj is None:
                callback(filepath, None, pbar)
                continue

            vid_obj = vid_obj.auto_resize()
            print(f'{filepath} SCALE {scale}')
            np_frames = vid_obj.out_video

            face_image_map = FaceExtractor.faces_from_video(
                np_frames, rescale=1, filename=filepath,
                every_n_frames=every_n_frames, coords_scale=scale,
                fill_face_maps=fill_face_maps
            )

            callback(filepath, face_image_map, pbar)

    def callback(self, filepath, face_image_map, pbar=None):
        if face_image_map is None:
            return False

        name = filepath
        if '/' in filepath:
            name = name[name.rindex('/') + 1:]

        name = name[:name.index('.')]
        filename = f'{name}.mp4'
        print(f'NAME {name}')

        export_dir = 'datasets-local/mtcnn-faces'
        face_dir = f'{export_dir}/{name}'

        if not os.path.exists(export_dir):
            os.mkdir(export_dir)
        if not os.path.exists(face_dir):
            os.mkdir(face_dir)

        # print(f'FACE IMAGE MAP {face_image_map}')

        for face_no in face_image_map:
            faces = face_image_map[face_no]
            for i, frame_no in enumerate(faces):
                face = faces[frame_no]
                frame = face.image
                face_no = face.face_no
                frame_no = face.frame_no
                num_faces = face.num_faces
                coords = face.coords

                top, right, bottom, left = coords

                self.filename_log.append(filename)
                self.num_face_log.append(num_faces)
                self.frame_log.append(frame_no)
                self.face_log.append(face_no)

                self.top_log.append(top)
                self.left_log.append(left)
                self.right_log.append(right)
                self.bottom_log.append(bottom)

                im = Image.fromarray(frame)
                path = f'{face_dir}/{face_no}-{frame_no}.jpg'
                im.save(path)

    def extract_all(self, every_n_frames=20):
        self.filename_log = []
        self.num_face_log = []
        self.frame_log = []
        self.face_log = []

        self.top_log = []
        self.left_log = []
        self.right_log = []
        self.bottom_log = []

        dataset = datasets.Dataset(basedir='datasets')
        filenames = dataset.all_videos[:].tolist()
        filepaths = []

        for k in range(len(filenames)):
            filename = filenames[k]
            filepath = f'datasets/train/videos/{filename}'
            filepaths.append(filepath)

        start_time = time.perf_counter()
        # input(f'IN FILEPATHS {filepaths}')
        self.process_filepaths(
            filepaths, every_n_frames=every_n_frames,
            batch_size=16, callback=self.callback
        )

        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f'extract duration: {duration}')
        df = pd.DataFrame(data={
            'filename': self.filename_log,
            'num_faces': self.num_face_log,
            'face_no': self.face_log,
            'frame': self.frame_log,

            'top': self.top_log,
            'left': self.left_log,
            'right': self.right_log,
            'bottom': self.bottom_log
        })

        date_stamp = self.make_date_stamp()
        export_path = f'stats/mtcnn-detect-{date_stamp}.csv'
        df.to_csv(export_path, index=False)
        print(f'exported to {export_path}')