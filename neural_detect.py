import loader
import torch
import numpy as np
import pandas as pd
import os

from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from torchvision import datasets

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

filename = '0a4f0c8985297ed7.mp4'
# filename = '0a8e8e7b229fe1fc.mp4'
filepath = f'datasets/train/videos/{filename}'

video = loader.load_video(filepath, every_n_frames=20, scale=0.5)
print(f'length = {len(video.out_video)}')
image = video.out_video[10]

bboxes, confs = mtcnn.detect(video.out_video[10:20])
print(f'BBOX = {bboxes}')
box = bboxes[0][0]

pil_img = Image.fromarray(image)
img_draw = ImageDraw.Draw(pil_img)
left, top, right, bottom = box
print(f'BOX = {box}')
shape = [(left, top), (right, bottom)]
img_draw.rectangle(shape, outline='#AAFF00', width=10)
pil_img.show()

print(f'bbox = {bbox}')


