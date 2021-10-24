try:
    import ParentImport
    import datasets

    from network.models import model_selection
    from detect_from_video import predict_with_model

except ModuleNotFoundError:
    from . import ParentImport
    from .. import datasets

    from .network.models import model_selection
    from .detect_from_video import predict_with_model

import os
import numpy as np

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


class FaceRecords(object):
    def __init__(self):
        dt = datetime.datetime.now()
        self.stamp = dt.strftime('%Y%m%d-%H%M')[2:]

        self.filename_log = []
        self.face_log = []
        self.frame_log = []

        self.top_log = []
        self.left_log = []
        self.right_log = []
        self.bottom_log = []
        self.buffer_log = []

        self.x_scale_log = []
        self.y_scale_log = []
        self.num_face_log = []

    def add(
        self, filename, frame_no, face_no,
        top, left, right, bottom, buffer,
        x_scale, y_scale, num_faces
    ):
        self.filename_log.append(filename)
        self.frame_log.append(frame_no)
        self.face_log.append(face_no)

        self.top_log.append(top)
        self.left_log.append(left)
        self.right_log.append(right)
        self.bottom_log.append(bottom)
        self.buffer_log.append(buffer)

        self.x_scale_log.append(x_scale)
        self.y_scale_log.append(y_scale)
        self.num_face_log.append(num_faces)

    def export(self):
        path = f'csvs/sorted-detections-{self.stamp}.csv'
        df = pd.DataFrame(data={
            'filename': self.filename_log,
            'frame': self.frame_log, 'face': self.face_log,

            'top': self.top_log, 'left': self.left_log,
            'right': self.right_log, 'bottom': self.bottom_log,
            'buffer': self.buffer_log,

            'x_scale': self.x_scale_log, 'y_scale': self.y_scale_log,
            'num_faces': self.num_face_log
        })

        df.to_csv(path, index=False)
        print(f'sorted detections saved to: {path}')


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

    @classmethod
    def faces_from_video(
        cls, np_frames, filename, rescale, export_size=256
    ):
        num_faces, face_mapping = cls.fill_face_maps(np_frames, 1)
        faces_df = cls.face_map_to_df(
            np_frames, num_faces, face_mapping,
            every_n_frames=1, current_rescale=1,
            filename=filename
        )

        group_face_frames = cls.collate_faces(faces_df, num_faces)
        sorted_face_frames = cls.sort_face_frames(
            group_face_frames, num_faces=num_faces
        )

        excluded_faces = cls.exclude_faces(sorted_face_frames)
        face_image_map, shift_left = {}, 0

        for face_no in range(num_faces):
            if face_no in excluded_faces:
                shift_left += 1
                continue

            face_images = []
            shifted_face_no = max(face_no - shift_left, 0)
            face_image_map[shifted_face_no] = face_images
            face_rows = sorted_face_frames[face_no]

            for i, row in enumerate(face_rows):
                frame = np_frames[i]
                top, left, right, bottom = cls.extract_coords(
                    i, face_rows, every_n_frames=1
                )

                face_crop, ratio = cls.get_square_face(
                    frame, top, left, right, bottom,
                    export_size=export_size, rescale=rescale,
                    rescale_ratios=None
                )

                face_images.append(face_crop)

        return face_image_map

    @staticmethod
    def sort_face_frames(face_frames, num_faces):
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

        return sorted_face_frames

    @staticmethod
    def face_map_to_df(
        np_frames, max_faces, face_mapping, every_n_frames,
        current_rescale, filename
    ):
        face_coords_df = pd.DataFrame(columns=[
            'filename', 'frames', 'face_no', 'num_faces',
            'top', 'right', 'bottom', 'left'
        ])

        detections, column = 0, 0

        for frame_no in face_mapping:
            image = np_frames[frame_no // every_n_frames]
            face_locations = face_recognition.face_locations(image)
            faces = len(face_locations)

            if faces > 0:
                detections += 1

            max_faces = max(max_faces, faces)
            num_faces = len(face_locations)

            for face_no in range(len(face_locations)):
                face_location = face_locations[face_no]
                top, right, bottom, left = face_location

                top = int(top / current_rescale)
                right = int(right / current_rescale)
                bottom = int(bottom / current_rescale)
                left = int(left / current_rescale)

                face_coords_df.loc[column] = [
                    filename, frame_no, face_no, num_faces,
                    top, right, bottom, left
                ]

                column += 1

        return face_coords_df

    @staticmethod
    def fill_face_maps(np_frames, interval):
        face_mapping = {}
        max_faces = 0

        for i in range(len(np_frames)):
            image = np_frames[i]
            frame_no = interval * i
            face_locations = face_recognition.face_locations(image)
            face_mapping[frame_no] = face_locations

            num_faces = len(face_locations)
            # print('faces', num_faces, max_faces)
            max_faces = max(max_faces, num_faces)

        return max_faces, face_mapping

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

    @staticmethod
    def square_coords(
        top, left, right, bottom,
        x_scale, y_scale, rescale, image, shrink=False
    ):
        img_width = image.shape[1] / rescale
        img_height = image.shape[0] / rescale
        x_clip = img_width * (1 - 1.0 / x_scale) / 2
        y_clip = img_height * (1 - 1.0 / y_scale) / 2

        top = (top - y_clip) * y_scale
        bottom = (bottom - y_clip) * y_scale
        left = (left - x_clip) * x_scale
        right = (right - x_clip) * x_scale

        width = right - left
        height = bottom - top
        diff = abs(width - height)

        if (bottom - top) > (right - left):
            left = max(left - diff / 2, 0)
            right = min(right + diff / 2, img_width)
            distance = (bottom - top) - (right - left)

            if distance > 0:
                if left == 0:
                    right += distance
                elif right == img_width:
                    left -= distance

        elif (bottom - top) < (right - left):
            top = max(top - diff / 2, 0)
            bottom = min(bottom + diff / 2, img_height)
            distance = (right - left) - (bottom - top)

            if distance > 0:
                if top == 0:
                    bottom += distance
                elif bottom == img_height:
                    top -= distance

        top = max(top, 0)
        bottom = min(bottom, img_height)
        left = max(left, 0)
        right = min(right, img_width)

        width = right - left
        height = bottom - top
        clip = abs(width - height)
        # print('SHRINK', width, height, x_scale, y_scale)

        if shrink and (clip != 0):
            if width > height:
                top -= clip // 2
                bottom += clip // 2
            else:
                left -= clip // 2
                right += clip // 2

        top = (top / y_scale) + y_clip
        bottom = (bottom / y_scale) + y_clip
        left = (left / x_scale) + x_clip
        right = (right / x_scale) + x_clip
        return top, left, right, bottom

    @classmethod
    def get_square_face(
        cls, frame, top, left, right, bottom,
        rescale_ratios=None, export_size=256, rescale=1
    ):
        img_width = frame.shape[1]
        img_height = frame.shape[0]

        area = (bottom - top) * (right - left)
        area_root = area ** 0.5
        buffer = int(area_root // 7)

        x_scale, y_scale = 1, 1
        if rescale_ratios is not None:
            x_scale, y_scale = rescale_ratios

        b_top = max(top - buffer / y_scale, 0)
        b_left = max(left - buffer / x_scale, 0)
        b_right = min(right + buffer / x_scale, img_width)
        b_bottom = min(bottom + buffer / y_scale, img_height)

        # get square coordinates
        s_top, s_left, s_right, s_bottom = cls.square_coords(
            top, left, right, bottom,
            x_scale, y_scale, rescale, frame
        )
        # find largest bounding box between square, buffer
        f_top, f_left, f_right, f_bottom = (
            min(b_top, s_top), min(b_left, s_left),
            max(b_right, s_right), max(b_bottom, s_bottom)
        )
        # get square coordinates of largest bounding box
        f_top, f_left, f_right, f_bottom = cls.square_coords(
            f_top, f_left, f_right, f_bottom,
            x_scale, y_scale, rescale, frame, shrink=True
        )

        f_top = int(f_top * rescale)
        f_left = int(f_left * rescale)
        f_right = int(f_right * rescale)
        f_bottom = int(f_bottom * rescale)
        face_crop = frame[f_top:f_bottom, f_left:f_right]
        # print('SCALE', x_scale, y_scale)

        # new_width = int(face_crop.shape[1] * x_scale)
        # new_height = int(face_crop.shape[0] * y_scale)
        scaled_width = int((f_right - f_left) * x_scale)
        scaled_height = int((f_bottom - f_top) * y_scale)
        ratio = scaled_width / scaled_height
        """
        face_crop = cv2.resize(
            face_crop, (scaled_width, scaled_height)
        )
        """
        dimensions = (scaled_width, scaled_height)
        face_crop = cv2.resize(face_crop, dimensions)
        width, height = face_crop.shape[1], face_crop.shape[0]

        # print('FF', face_crop.shape, width, height)
        clip = abs(width - height) // 2
        # print(clip, width > height)

        if clip != 0:
            if width > height:
                face_crop = face_crop[:, clip:-clip]
            else:
                face_crop = face_crop[clip:-clip, :]

        dimensions = (export_size, export_size)
        # print('NEW SHAPE', face_crop.shape)
        face_crop = cv2.resize(face_crop, dimensions)
        return face_crop, ratio

    def export_face_frames(
        self, face_frames, num_faces, rescale_ratios,
        np_frames, base_dir, face_records, filename,
        excluded_faces=(), export_size=256, rescale=None
    ):
        if rescale is None:
            rescale = self.rescale

        shift_left = 0
        bad_ratios = []

        for face_no in range(num_faces):
            if face_no in excluded_faces:
                shift_left += 1
                continue

            #  print(f'FRAME NO {i} {np_frames.shape}')
            face_rows = face_frames[face_no]

            # print('rows', face_rows)

            for i, row in enumerate(face_rows):
                frame_no = row['frames']
                index = frame_no // self.every_n_frames
                frame = np_frames[index]
                # print('ROW', row)

                top, left, right, bottom = self.extract_coords(
                    i, face_rows, self.every_n_frames
                )

                if top == float('inf'):
                    continue

                face_crop, ratio = self.get_square_face(
                    frame, top, left, right, bottom,
                    rescale_ratios=rescale_ratios,
                    export_size=export_size, rescale=rescale
                )

                if (ratio > 1.05) or (ratio < 0.95):
                    bad_ratios.append((face_no, ratio))

                face_records.add(
                    filename=filename, frame_no=frame_no,
                    face_no=face_no, buffer=buffer,
                    top=top, left=left, right=right, bottom=bottom,
                    x_scale=x_scale, y_scale=y_scale,
                    num_faces=num_faces
                )

                im = Image.fromarray(face_crop)
                shifted_face_no = max(face_no - shift_left, 0)
                path = f'{base_dir}/{shifted_face_no}-{frame_no}.jpg'
                im.save(path)

        if len(bad_ratios) > 0:
            print('POORLY SIZED', filename, bad_ratios)

    @staticmethod
    def exclude_faces(sorted_face_frames, min_frames=5):
        num_faces = len(sorted_face_frames)
        face_nos = tuple(sorted_face_frames.keys())
        excluded_faces = []

        for face_no in face_nos:
            if len(sorted_face_frames) == 1:
                break

            frames = sorted_face_frames[face_no]
            if len(frames) < min_frames:
                excluded_faces.append(face_no)

        if len(excluded_faces) == num_faces:
            best_index, max_frames = 0, float('-inf')

            for k, face_no in enumerate(face_nos):
                frames = sorted_face_frames[face_no]
                if len(frames) > max_frames:
                    max_frames = len(frames)
                    best_index = k

            del excluded_faces[best_index]

        return excluded_faces

    def extract_faces(
        self, filenames=None, export_dir='../datasets-local/faces',
        pre_resize=False, export_df=False, export_size=256
    ):
        if filenames is None:
            filenames = self.dataset.all_videos

        face_records = FaceRecords()
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

            sorted_face_frames = self.sort_face_frames(
                face_frames, num_faces=num_faces
            )
            excluded_faces = self.exclude_faces(sorted_face_frames)

            self.export_face_frames(
                sorted_face_frames, num_faces, rescale_ratios,
                np_frames, base_dir, excluded_faces=excluded_faces,
                filename=filename, face_records=face_records,
                rescale=self.rescale, export_size=export_size
            )

        # base_faces.to_csv(output_path, index=False)
        # print(f'SAVED TO {output_path}')
        print(f'INVALID VIDEOS', self.invalid_videos)
        print(f'FACELESS VIDEOS', faceless_videos)

        if export_df:
            face_records.export()


