import loader
import torch
import numpy as np
import pandas as pd
import os

from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision import datasets

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

filename = '0a8e8e7b229fe1fc.mp4'
filepath = f'~/projects/AISG/datasets/train/videos/{filename}'

video = loader.load_video(filepath, every_n_frames=20)
image = video.out_video[0]

bbox = mtcnn.detect(image)
print(f'bbox = {bbox}')


