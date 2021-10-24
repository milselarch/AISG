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


def run():
    pbar = tqdm(range(num_videos))

    for k in pbar:
        filename = dataset.all_videos[k]
        pbar.set_description(f"Processing {filename}")

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

        np_frames = vid_obj.out_video
        num_frames = len(np_frames)
        video_frame_rows = base_faces[
            base_faces['filename'] == filename
        ]

        # print('FRAME ROWS', video_frame_rows)
        # print('FRAMES', frames_column)
        # print(240 in frames_column)

        for i in range(num_frames):
            #  print(f'FRAME NO {i} {np_frames.shape}')
            frame_no = every_n_frames * i
            frane = np_frames[i]
            face_rows = video_frame_rows[
                video_frame_rows["frames"] == frame_no
            ]

            # print('rows', face_rows)

            for index in face_rows.index:
                row = face_rows.loc[index]
                # print('ROW', row)

                top = int(row["top"] * rescale)
                left = int(row["left"] * rescale)
                right = int(row["right"] * rescale)
                bottom = int(row["bottom"] * rescale)

                coords = (top, left, right, bottom)
                face_crop = frane[top:bottom, left:right]

                try:
                    _, frame_output = predict_with_model(
                        face_crop, model, cuda=cuda
                    )
                except cv2.error as e:
                    print('FACE CROP', face_crop)
                    print('SCALE', rescale)
                    print('ROW FAIL', row)
                    print(f'SHAPE {face_crop.shape}')
                    print(f'COORDS {coords}')
                    print('FILENAME', filename)
                    raise e

                predictions = frame_output.cpu().detach().numpy()
                prediction = predictions[0][1]
                pred_column = base_faces['prediction']
                pred_column[index] = prediction
                # print(f'PREDICTION {prediction}')
                # print(base_faces.loc[index])
                # input('>>> ')

    base_faces.to_csv(output_path, index=False)
    print(f'SAVED TO {output_path}')
    print(f'INVALID VIDEOS', invalid_videos)


if __name__ == '__main__':
    profile = cProfile.Profile()
    profile.enable()
    run()
    profile.disable()
    profile.dump_stats(profile_path)