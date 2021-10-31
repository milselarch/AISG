import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
    raise Exception("Must be using >= Python 3.2")

from os import listdir, path

if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
    raise FileNotFoundError(
        'Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
		before running this script!'
    )

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
import audio
from hparams import hparams as hp

import face_detection

NGPU = 1

fa = [
    face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, flip_input=False,
        device='cuda:{}'.format(id)
    ) for id in range(NGPU)
]

template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'


# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'

def process_video_file(
    vfile, gpu_id=0,
    preprocessed_root='lrs2_preprocessed',
    batch_size=1
):
    video_stream = cv2.VideoCapture(vfile)

    frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)

    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = path.join(preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    batches = [
        frames[i:i + batch_size]
        for i in range(0, len(frames), batch_size)
    ]

    i = -1
    for fb in tqdm(batches):
        preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))

        for j, f in enumerate(preds):
            i += 1
            if f is None:
                continue

            x1, y1, x2, y2 = f
            cv2.imwrite(
                path.join(fulldir, '{}.jpg'.format(i)),
                fb[j][y1:y2, x1:x2]
            )


process_video_file(
    '../datasets/train/videos/0ae95b34e9481b4f.mp4'
)