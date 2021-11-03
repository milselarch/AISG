try:
    import datasets
    from DeepfakeDetection.FaceExtractor import FaceExtractor
except ModuleNotFoundError:
    from . import datasets
    from .DeepfakeDetection.FaceExtractor import FaceExtractor

import loader_v2 as loader
import torch
import numpy as np
import pandas as pd
import functools
import time
import cv2
import os
import gc

from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from datetime import datetime
from tqdm.auto import tqdm

class NeuralFaceExtract(object):
    def __init__(self):
        self.export_dir = 'datasets-local/mtcnn-faces'

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=50,
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

    def extract_faces(
        self, frames, batch_size, interval,
        skip_detect=None, export_size=256
    ):
        sub_frames = []
        sub_frame_nos = []
        last_frame_no = 0

        for k in range(len(frames)):
            frame_no = interval * k
            frame = frames[k]

            if skip_detect is None:
                sub_frames.append(frame)
                sub_frame_nos.append(frame_no)
            elif frame_no - last_frame_no >= skip_detect:
                sub_frames.append(frame)
                sub_frame_nos.append(frame_no)
                last_frame_no = frame_no

        frame_face_boxes, index = [], 0
        face_confs = []

        while index < len(sub_frames):
            # print(index, len(sub_frames))
            end_index = index + batch_size
            batch = sub_frames[index:end_index]
            np_batch = frames.resolve_batch(batch)
            bboxes, bconfs = self.mtcnn.detect(np_batch)

            face_confs.extend(bconfs)
            frame_face_boxes.extend(bboxes)
            index = end_index

        iterator = zip(sub_frame_nos, frame_face_boxes, face_confs)
        detect_map = {}

        for frame_no, bbox, bconf in iterator:
            detect_map[frame_no] = (bbox, bconf)

        prev_bboxes, prev_bconfs = None, None
        total_face_frame_boxes = []
        total_face_confs = []
        detect_frame_nos = []
        face_crop_map = {}

        last_frame_no = 0

        for k in range(len(frames)):
            frame = frames[k]
            # if we can directly optimise not needing
            # np_frame, theres room for improvement
            np_frame = frame.to_numpy()
            frame_no = interval * k

            if frame_no in detect_map:
                prev_bboxes, prev_bconfs = detect_map[frame_no]
                detect_frame_nos.append(frame_no)
            elif last_frame_no - frame_no > skip_detect:
                prev_bboxes, prev_bconfs = None, None
                # detect_frame_nos.append(frame_no)

            if prev_bboxes is not None:
                # extract face images
                for k, bbox in enumerate(prev_bboxes):
                    bbox = bbox.astype(int)
                    bbox = np.clip(bbox, a_max=999999, a_min=0)
                    prev_bboxes[k] = bbox

                    bbox_tuple = tuple(bbox.tolist())
                    key = (frame_no, bbox_tuple)
                    left, top, right, bottom = bbox

                    extraction = FaceExtractor.get_square_face(
                        np_frame, top, left, right, bottom,
                        rescale_ratios=None, rescale=1,
                        export_size=export_size
                    )
                    face_crop_map[key] = extraction

            total_face_frame_boxes.append(prev_bboxes)
            total_face_confs.append(prev_bconfs)

        return (
            total_face_frame_boxes, total_face_confs,
            detect_frame_nos, face_crop_map
        )

    def fill_face_maps(
        self, frames, interval, batch_size, skip_detect=None,
        export_size=256
    ):
        extract_result = self.extract_faces(
            frames, batch_size=batch_size, interval=interval,
            skip_detect=skip_detect, export_size=export_size
        )

        frame_face_boxes = extract_result[0]
        face_confs = extract_result[1]
        detect_frame_nos = extract_result[2]
        face_crop_map = extract_result[3]

        max_faces, face_mapping = 0, {}
        # assert len(frame_face_boxes) == len(np_frames)
        # print(frame_face_boxes)

        for k in range(len(frame_face_boxes)):
            frame_no = interval * k
            face_locations = []
            face_mapping[frame_no] = face_locations

            bboxes = frame_face_boxes[k]
            bconfs = face_confs[k]

            if bboxes is None:
                # print(f'BBOXES IS NONE')
                continue

            assert bconfs is not None
            assert bboxes is not None
            for bbox, bconf in zip(bboxes, bconfs):
                if bconf < 0.991:
                    # face detection confidence threshold
                    continue

                bbox = bbox.astype(int)
                # bbox = np.clip(bbox, a_max=999999, a_min=0)
                assert min(bbox) >= 0
                # print(f'BBOX {k} {frame_no} {bbox} {bconf}')
                left, top, right, bottom = bbox

                assert right > left
                assert bottom > top
                face_locations.append((
                    top, right, bottom, left
                ))

            num_faces = len(face_locations)
            max_faces = max(max_faces, num_faces)

        return (
            max_faces, face_mapping, detect_frame_nos,
            face_crop_map
        )

    def process_filepaths(
        self, filepaths, callback=lambda *args, **kwargs: None,
        every_n_frames=20, batch_size=16, base_dir=None,
        export_size=256, skip_detect=None,
        img_filter=lambda x, f: x
    ):
        """
        def fill_face_maps(frames, interval, skip_detect=None):
            return self.fill_face_maps(
                frames, interval, batch_size=batch_size,
                skip_detect=skip_detect
            )
        """
        pbar = tqdm(filepaths)

        for filepath in pbar:
            pbar.set_description(filepath)

            if base_dir is not None:
                filepath = f'{base_dir}/{filepath}'

            video_cap = cv2.VideoCapture(filepath)
            width_in = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_in = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            scale = 0.5
            if min(width_in, height_in) < 700:
                scale = 1

            vid_obj = loader.load_video(
                video_cap, every_n_frames=every_n_frames,
                scale=scale, reset_index=False
            )

            if vid_obj is None:
                callback(filepath, None, pbar)
                continue

            # vid_obj = vid_obj.auto_resize()
            vid_obj.auto_resize_inplace()
            print(f'{filepath} SCALE {scale}')
            # np_frames = vid_obj.out_video

            face_image_map = FaceExtractor.faces_from_video(
                vid_obj, rescale=1, filename=filepath,
                every_n_frames=every_n_frames, coords_scale=scale,
                export_size=export_size, skip_detect=skip_detect,
                fill_face_maps=functools.partial(
                    self.fill_face_maps,  batch_size=batch_size,
                    skip_detect=skip_detect, export_size=export_size
                )
            )

            callback(
                filepath=filepath, face_image_map=face_image_map,
                pbar=pbar, img_filter=img_filter
            )

            vid_obj.release()
            del video_cap
            del vid_obj

    def callback(
        self, filepath, face_image_map, pbar=None,
        img_filter=lambda x, f: x
    ):
        if face_image_map is None:
            print(f'VIDEO LOAD FAILED {filepath}')
            return False

        name = filepath
        if '/' in filepath:
            name = name[name.rindex('/') + 1:]

        name = name[:name.index('.')]
        filename = f'{name}.mp4'
        # print(f'NAME {name}')

        face_dir = f'{self.export_dir}/{name}'

        if not os.path.exists(self.export_dir):
            os.mkdir(self.export_dir)
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

                # top, right, bottom, left = coords
                top, left, right, bottom = coords

                if face.detected:
                    self.filename_log.append(filename)
                    self.num_face_log.append(num_faces)
                    self.frame_log.append(frame_no)
                    self.face_log.append(face_no)

                    self.top_log.append(top)
                    self.left_log.append(left)
                    self.right_log.append(right)
                    self.bottom_log.append(bottom)

                frame = img_filter(frame, face)
                im = Image.fromarray(frame)
                path = f'{face_dir}/{face_no}-{frame_no}.jpg'
                im.save(path)

        del face_image_map
        gc.collect()

    def extract_all(
        self, filenames=None, every_n_frames=20,
        export_size=256, skip_detect=None,
        img_filter=lambda x, f: x
    ):
        self.filename_log = []
        self.num_face_log = []
        self.frame_log = []
        self.face_log = []

        self.top_log = []
        self.left_log = []
        self.right_log = []
        self.bottom_log = []

        dataset = datasets.Dataset(basedir='datasets')
        if filenames is None:
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
            batch_size=16, callback=self.callback,
            export_size=export_size, skip_detect=skip_detect,
            img_filter=img_filter
        )

        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f'extract duration: {duration}')
        df = pd.DataFrame(data={
            'filename': self.filename_log,
            'num_faces': self.num_face_log,
            'face': self.face_log,
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
