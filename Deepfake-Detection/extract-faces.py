import os

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

dt = datetime.datetime.now()
stamp = dt.strftime('%Y%m%d-%H%M%S')
dataset = datasets.Dataset(basedir='../datasets')
# dataset.all_videos

# print(dataframe.head())
print(len(dataset.train_videos), dataset.train_videos[:5])
print(len(dataset.test_videos), dataset.test_videos[:5])

max_face_mapping = {}
num_videos = len(dataset.all_videos)
# num_videos = 10

scale_down = 2
every_n_frames = 20
n = every_n_frames
cuda = True

assert type(scale_down) is int
assert scale_down >= 1
rescale = 1 / scale_down

output_path = f'../stats/detections-{stamp}.csv'
profile_path = f'../stats/extract-{stamp}.profile'
base_filename = "face_map_stats-4-20-20210902-164550.csv"
base_faces = pd.read_csv(f'../stats/{base_filename}')
base_faces['prediction'] = None
invalid_videos = []

print(f'rescale is {rescale}')
print(f'CSV filename is {output_path}')

model = model_selection(
    modelname='xception', num_out_classes=2, dropout=0.5
)

model_path = './pretrained_model/deepfake_c0_xception.pkl'
model.load_state_dict(torch.load(model_path))
model = model.cuda() if cuda else model

def extract_coords(
    index, np_frames, video_frame_rows,
    every_n_frames, face_no
):
    min_top = float('inf')
    min_left = float('inf')
    max_right = float('-inf')
    max_bottom = float('-inf')

    for offset in (-1, 0, 1):
        i = index + offset

        if (i == -1) or (i == len(np_frames)):
            continue

        frame_no = every_n_frames * i
        face_rows = video_frame_rows[
            video_frame_rows["frames"] == frame_no
        ]

        for index in face_rows.index:
            if face_no == index:
                continue

            row = face_rows.loc[index]

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

def extract_faces():
    pbar = tqdm(range(num_videos))

    for k in pbar:
        filename = dataset.all_videos[k]
        name = filename[:filename.index('.')]
        base_dir = f'../datasets-local/faces/{name}'
        pbar.set_description(f"Processing {filename}")

        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        try:
            vid_obj = dataset.read_video(
                filename, every_n_frames=every_n_frames,
                scale=rescale
            )
        except datasets.FailedVideoRead:
            print(f'FILENAME LOAD FAILED {filename}')
            continue
        except ValueError as e:
            invalid_videos.append(filename)
            print(f'VALUE ERROR {filename} {k}')
            raise e

        vid_obj = vid_obj.auto_resize()
        np_frames = vid_obj.out_video
        num_frames = len(np_frames)
        video_frame_rows = base_faces[
            base_faces['filename'] == filename
        ]

        # print('FRAME ROWS', video_frame_rows)
        # print('FRAMES', frames_column)
        # print(240 in frames_column)

        excluded_face_nos = []
        num_faces = video_frame_rows['num_faces'].to_numpy()[0]
        for face_no in range(num_faces):
            frames = video_frame_rows[
                video_frame_rows["face_no"] == face_no
            ]

            if len(frames) < 5:
                excluded_face_nos.append(face_no)

        for i in range(num_frames):
            #  print(f'FRAME NO {i} {np_frames.shape}')
            frame_no = every_n_frames * i
            frane = np_frames[i]
            face_rows = video_frame_rows[
                video_frame_rows["frames"] == frame_no
            ]

            # print('rows', face_rows)

            for face_no in face_rows.index:
                if face_no in excluded_face_nos:
                    continue

                row = face_rows.loc[face_no]
                # print('ROW', row)

                top, left, right, bottom = extract_coords(
                    i, np_frames, video_frame_rows,
                    every_n_frames, face_no
                )

                if top == float('inf'):
                    continue

                area = (bottom - top) * (right - left)
                area_root = area ** 0.5
                buffer = int(area_root // 7)

                b_top = int(rescale * max(top - buffer, 0))
                b_left = int(rescale * max(left - buffer, 0))
                b_right = int(rescale * (right + buffer))
                b_bottom = int(rescale * (bottom + 1.4 * buffer))

                face_crop = frane[b_top:b_bottom, b_left:b_right]
                im = Image.fromarray(face_crop)
                path = f'{base_dir}/{face_no}-{frame_no}.jpg'
                im.save(path)

    # base_faces.to_csv(output_path, index=False)
    # print(f'SAVED TO {output_path}')
    print(f'INVALID VIDEOS', invalid_videos)


if __name__ == '__main__':
    profile = cProfile.Profile()
    profile.enable()
    extract_faces()
    profile.disable()
    profile.dump_stats(profile_path)