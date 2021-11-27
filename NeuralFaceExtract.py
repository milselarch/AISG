try:
    import datasets
    import loader_v2 as loader

    from FaceImageMap import FaceImageMap
    from DeepfakeDetection.FaceExtractor import FaceExtractor
except ModuleNotFoundError:
    from . import datasets
    from . import loader_v2 as loader

    from .FaceImageMap import FaceImageMap
    from .DeepfakeDetection.FaceExtractor import FaceExtractor

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

class ImageMap(object):
    def __init__(
        self, face_crop, mouth_crop, landmark=None,
        frame_no=None, ratio=None, blended=False
    ):
        self.face_crop = face_crop
        self.ratio = ratio

        self.mouth_crop = mouth_crop
        self.landmark = landmark
        self.frame_no = frame_no
        self.blended = blended


class NeuralFaceExtract(object):
    def __init__(self, use_cuda=True):
        self.export_dir = 'datasets-local/mtcnn-faces'
        
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
            
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

    @staticmethod
    def min_overlap_p(bbox1, bbox2):
        left1, top1, right1, bottom1 = bbox1
        left2, top2, right2, bottom2 = bbox2
        assert (bottom1 >= top1) and (right1 >= left1)
        assert (bottom2 >= top2) and (right2 >= left2)

        overlap_left = max(left1, left2)
        overlap_right = min(right1, right2)
        overlap_top = max(top1, top2)
        overlap_bottom = min(bottom1, bottom2)

        overlap_width = overlap_right - overlap_left
        overlap_height = overlap_bottom - overlap_top
        overlap_width = max(overlap_width, 0)
        overlap_height = max(overlap_height, 0)

        overlap_area = overlap_width * overlap_height
        area1 = (right1 - left1) * (bottom1 - top1)
        area2 = (right2 - left2) * (bottom2 - top2)
        overlap_p = overlap_area / max(area1, area2)
        return overlap_p

    def extract_faces(
        self, frames, batch_size, interval,
        skip_detect=None, ignore_detect=None, export_size=256,
        conf_threshold=0.991, make_next_detect_map=True,
        displace_next=2
    ):
        sub_frames = []
        sub_frame_nos = []
        last_frame_no = 0

        if ignore_detect is None:
            ignore_detect = interval

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
        face_confs, face_landmarks = [], []

        while index < len(sub_frames):
            # print(index, len(sub_frames))
            end_index = index + batch_size
            batch = sub_frames[index:end_index]
            np_batch = frames.resolve_batch(batch)
            b_boxes, b_confs, b_landmarks = self.mtcnn.detect(
                np_batch, landmarks=True
            )

            face_confs.extend(b_confs)
            frame_face_boxes.extend(b_boxes)
            face_landmarks.extend(b_landmarks)
            index = end_index

        iterator = list(zip(
            sub_frame_nos, frame_face_boxes, face_confs,
            face_landmarks
        ))

        detect_map, next_detect_map = {}, {}

        for k, item in enumerate(iterator):
            frame_no, bboxes, bconfs, landmarks = item
            detect_map[frame_no] = (bboxes, bconfs, landmarks)
            if not make_next_detect_map:
                continue

            try:
                next_item = iterator[k+displace_next]
            except IndexError:
                continue

            next_frame_no = next_item[0]
            next_bboxes = next_item[1]
            next_bconfs = next_item[2]
            next_landmarks = next_item[3]

            clip_max = 999999
            if (bboxes is None) or (next_bboxes is None):
                continue

            for c_index, bbox_raw in enumerate(bboxes):
                bbox = bbox_raw.astype(int)
                bbox = np.clip(bbox, 0, clip_max)
                bbox_tuple = tuple(bbox.tolist())

                for n_index, next_bbox_raw in enumerate(next_bboxes):
                    next_bbox = next_bbox_raw.astype(int)
                    next_bbox = np.clip(next_bbox, 0, clip_max)

                    next_conf = next_bconfs[n_index]
                    if next_conf < conf_threshold:
                        continue

                    overlap_p = self.min_overlap_p(bbox, next_bbox)
                    if overlap_p < 0.6:
                        continue

                    next_landmark = next_landmarks[n_index]
                    key = (frame_no, bbox_tuple)
                    next_detect_map[key] = (
                        next_frame_no, next_bbox, next_landmark
                    )
                
        return self._extract_faces_from_map(
            frames, detect_map, export_size=export_size,
            interval=interval, next_detect_map=next_detect_map,
            skip_detect=skip_detect, ignore_detect=ignore_detect
        )

    def _extract_faces_from_map(
        self, frames, detect_map, interval, skip_detect,
        ignore_detect, export_size=256, next_detect_map=None
    ):
        if next_detect_map is None:
            next_detect_map = {}

        prev_landmarks = None
        prev_bboxes = None
        prev_bconfs = None

        total_face_frame_boxes = []
        total_face_confs = []
        detect_frame_nos = []
        face_crop_map = {}

        last_detect_frame_no = 0

        for k in range(len(frames)):
            frame = frames[k]
            # if we can directly optimise not needing
            # np_frame, theres room for improvement
            frame_no = interval * k
            np_frame = None

            if frame_no in detect_map:
                info = detect_map[frame_no]
                prev_bboxes, prev_bconfs, prev_landmarks = info
                detect_frame_nos.append(frame_no)
                last_detect_frame_no = frame_no
            elif frame_no - last_detect_frame_no >= skip_detect:
                prev_bboxes, prev_bconfs = None, None
                # detect_frame_nos.append(frame_no)
            elif frame_no - last_detect_frame_no >= ignore_detect:
                prev_bboxes, prev_bconfs = None, None

            if prev_bboxes is not None:
                assert len(prev_bboxes) == len(prev_landmarks)

                # extract face images
                for i, bbox in enumerate(prev_bboxes):
                    if np_frame is None:
                        np_frame = frame.to_numpy()

                    bbox = bbox.astype(int)
                    bbox = np.clip(bbox, 0, 999999)
                    bbox_tuple = tuple(bbox.tolist())

                    key = (frame_no, bbox_tuple)
                    detect_key = (last_detect_frame_no, bbox_tuple)
                    landmark = prev_landmarks[i]
                    prev_bboxes[i] = bbox

                    if detect_key in next_detect_map:
                        next_detect_info = next_detect_map[detect_key]
                        detect_frame_no = next_detect_info[0]
                        next_bbox = next_detect_info[1]
                        next_landmark = next_detect_info[2]
                        min_lip_width = self.get_lip_width(
                            next_landmark
                        )

                        frames_passed = frame_no - last_detect_frame_no
                        dist = detect_frame_no - last_detect_frame_no
                        prog = frames_passed / dist
                        blended = True

                        crop_bbox, crop_landmark = (
                            prog * next_bbox + (1.-prog) * bbox,
                            prog * next_landmark + (1-prog) * landmark
                        )
                    else:
                        blended = False
                        min_lip_width = None
                        crop_landmark = landmark
                        crop_bbox = bbox

                    crop_bbox = crop_bbox.astype(int)
                    crop_bbox = np.clip(crop_bbox, 0, 999999)
                    left, top, right, bottom = crop_bbox

                    face_crop, ratio = FaceExtractor.get_square_face(
                        np_frame, top, left, right, bottom,
                        rescale_ratios=None, rescale=1,
                        export_size=export_size
                    )
                    mouth_crop = self.extract_mouth_crop(
                        np_frame, crop_landmark,
                        min_lip_width=min_lip_width,
                        export_size=export_size
                    )
                    face_crop_map[key] = ImageMap(
                        face_crop=face_crop, mouth_crop=mouth_crop,
                        landmark=crop_landmark, frame_no=frame_no,
                        ratio=ratio, blended=blended
                    )

            total_face_frame_boxes.append(prev_bboxes)
            total_face_confs.append(prev_bconfs)

        return (
            total_face_frame_boxes, total_face_confs,
            detect_frame_nos, face_crop_map
        )

    @staticmethod
    def get_lip_width(landmark):
        landmark = np.clip(landmark, a_max=999999, a_min=0)
        left_lip_coord, right_lip_coord = landmark[-2:]
        left_lip_x, left_lip_y = left_lip_coord
        right_lip_x, right_lip_y = right_lip_coord
        lip_width = right_lip_x - left_lip_x
        return lip_width

    @classmethod
    def extract_mouth_crop(
        cls, np_frame, landmark, export_size,
        crop_buffer_ratio=0.6, min_lip_width=None
    ):
        lip_width = cls.get_lip_width(landmark)
        if min_lip_width is not None:
            lip_width = max(lip_width, min_lip_width)

        lip_landmark = np.clip(landmark, a_max=999999, a_min=0)
        left_lip_coord, right_lip_coord = lip_landmark[-2:]
        left_lip_x, left_lip_y = left_lip_coord
        right_lip_x, right_lip_y = right_lip_coord

        midpoint_x = (left_lip_x + right_lip_x) / 2.
        midpoint_y = (left_lip_y + right_lip_y) / 2.

        crop_width = lip_width * (1. + crop_buffer_ratio)
        crop_width = max(crop_width, 10)

        left = midpoint_x - crop_width / 2.
        right = midpoint_x + crop_width / 2.
        top = midpoint_y - crop_width / 2.
        bottom = midpoint_y + crop_width / 2.

        try:
            mouth_crop, ratio = FaceExtractor.get_square_face(
                np_frame, top, left, right, bottom,
                rescale_ratios=None, rescale=1,
                export_size=export_size
            )
        except cv2.error as e:
            print('SQUARE EXTRACT FAIL')
            raise e

        return mouth_crop

    def fill_face_maps(
        self, frames, interval, batch_size, skip_detect=None,
        ignore_detect=None, export_size=256, make_next_detect_map=True
    ):
        extract_result = self.extract_faces(
            frames, batch_size=batch_size, interval=interval,
            skip_detect=skip_detect, ignore_detect=ignore_detect,
            make_next_detect_map=make_next_detect_map,
            export_size=export_size
        )

        frame_face_boxes = extract_result[0]
        face_confs = extract_result[1]
        detect_frame_nos = extract_result[2]
        face_crop_map = extract_result[3]

        max_faces = 0
        face_mapping, landmark_map = {}, {}
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
            iterable = zip(bboxes, bconfs)

            for bbox, bconf in iterable:
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

    def process_filepath(
        self, filepath, every_n_frames=20, batch_size=16,
        base_dir=None, export_size=256, skip_detect=None,
        ignore_detect=None, pbar=None, make_next_detect_map=True
    ):
        name = os.path.basename(filepath)
        if pbar is not None:
            pbar.set_description(name)

        if base_dir is not None:
            filepath = f'{base_dir}/{filepath}'

        video_cap = cv2.VideoCapture(filepath)
        width_in = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_in = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        scale = 0.5
        if min(width_in, height_in) < 700:
            scale = 1

        return self.process_video(
            video_cap, every_n_frames=every_n_frames,
            batch_size=batch_size, export_size=export_size,
            skip_detect=skip_detect, ignore_detect=ignore_detect,
            make_next_detect_map=make_next_detect_map,
            filepath=filepath
        )

    def process_video(
        self, video_cap, every_n_frames=20, batch_size=16,
        export_size=256, skip_detect=None, ignore_detect=None,
        scale=1, filepath='test', make_next_detect_map=True
    ):
        print(f'{filepath} SCALE {scale}')
        vid_obj = loader.load_video(
            video_cap, every_n_frames=every_n_frames,
            scale=scale, reset_index=False
        )

        if vid_obj is None:
            return None

        # vid_obj = vid_obj.auto_resize()
        vid_obj.auto_resize_inplace()
        vid_obj.force_no_resize()
        # np_frames = vid_obj.out_video

        face_image_map = FaceExtractor.faces_from_video(
            vid_obj, rescale=1, filename=filepath,
            every_n_frames=every_n_frames, coords_scale=scale,
            export_size=export_size, skip_detect=skip_detect,
            ignore_detect=ignore_detect,
            fill_face_maps=functools.partial(
                self.fill_face_maps, batch_size=batch_size,
                make_next_detect_map=make_next_detect_map,
                export_size=export_size
            )
        )

        face_image_map = FaceImageMap(
            face_image_map, fps=vid_obj.fps
        )

        return face_image_map

    def process_filepaths(
        self, filepaths, callback=lambda *args, **kwargs: None,
        every_n_frames=20, batch_size=16, base_dir=None,
        export_size=256, skip_detect=None, ignore_detect=None,
        img_filter=lambda x, f: x, make_next_detect_map=True
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
            face_image_map = self.process_filepath(
                filepath=filepath, every_n_frames=every_n_frames,
                base_dir=base_dir, export_size=export_size,
                skip_detect=skip_detect, ignore_detect=ignore_detect,
                pbar=pbar, make_next_detect_map=make_next_detect_map
            )

            callback(
                filepath=filepath, face_image_map=face_image_map,
                pbar=pbar, img_filter=img_filter
            )

            # vid_obj.release()
            # del vid_obj, video_cap
            # gc.collect()

    def save_image_map(
        self, filepath, face_image_map, pbar=None,
        img_filter=lambda x, f: x, save_mouth=False
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

                if save_mouth:
                    frame = face.mouth_image
                else:
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
        export_size=256, skip_detect=None, ignore_detect=None,
        img_filter=lambda x, f: x, basedir='datasets',
        video_base_dir='datasets/train/videos',
        save_mouth=False
    ):
        self.filename_log = []
        self.num_face_log = []
        self.frame_log = []
        self.face_log = []

        self.top_log = []
        self.left_log = []
        self.right_log = []
        self.bottom_log = []

        if filenames is None:
            dataset = datasets.Dataset(basedir=basedir)
            filenames = dataset.all_videos[:].tolist()

        save_image_map = functools.partial(
            self.save_image_map, save_mouth=save_mouth
        )

        filepaths = []
        for k in range(len(filenames)):
            filename = filenames[k]
            filepath = f'{video_base_dir}/{filename}'
            filepaths.append(filepath)

        start_time = time.perf_counter()
        # input(f'IN FILEPATHS {filepaths}')
        self.process_filepaths(
            filepaths, every_n_frames=every_n_frames,
            batch_size=16, callback=save_image_map,
            export_size=export_size, skip_detect=skip_detect,
            ignore_detect=ignore_detect, img_filter=img_filter
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
