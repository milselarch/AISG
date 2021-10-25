import loader
import torch
import numpy as np
import pandas as pd
import cv2
import os

from facenet_pytorch import MTCNN, InceptionResnetV1
from DeepfakeDetection.FaceExtractor import FaceExtractor
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from torchvision import datasets
from tqdm.auto import tqdm

class NeuralFaceExtract(object):
    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        self.mtcnn = MTCNN(
            image_size=320, margin=0, min_face_size=100,
            thresholds=[0.6, 0.7, 0.7], factor=0.709,
            post_process=True, device=self.device
        )

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

            for bbox, bconf in zip(bboxes, bconfs):
                if bconf < 0.9:
                    # face detection confidence threshold
                    continue

                bbox = bbox.astype(int)
                bbox = np.clip(bbox, a_max=999999, a_min=0)
                # print(f'BBOX {k} {bbox} {bconfs}')
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
        every_n_frames=20, batch_size=16
    ):
        def fill_face_maps(frames, interval):
            return self.fill_face_maps(
                frames, interval, batch_size=batch_size
            )

        for filepath in tqdm(filepaths):
            video_cap = cv2.VideoCapture(filepath)
            width_in = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_in = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            scale = 0.5
            if min(width_in, height_in) < 700:
                scale = 1

            vid_obj = loader.load_video(
                filepath, every_n_frames=20, scale=scale
            )

            vid_obj = vid_obj.auto_resize()
            print(f'{filepath} SCALE {scale}')
            np_frames = vid_obj.out_video

            face_image_map = FaceExtractor.faces_from_video(
                np_frames, rescale=1, filename=filepath,
                every_n_frames=every_n_frames, coords_scale=scale,
                fill_face_maps=fill_face_maps
            )

            callback(filepath, face_image_map)