import os
import numpy as np
import cv2
import random

import torch
from torchvision import transforms
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from models.syncnet_v2 import SyncNet_color

transform = transforms.Compose([
    transforms.Resize((128, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

syncnet_T = 5
model = SyncNet_color(syncnet_T)

def pil_loader(path: str, mirror_prob=0.5) -> Image.Image:
    with open(path, 'rb') as f:
        image = Image.open(f)
        if random.random() < mirror_prob:
            image = ImageOps.mirror(image)

        return image.convert('RGB')


# basedir = '../datasets-local/mtcnn-faces/3a3d68bceddb6dab'
basedir = '../datasets-local/mtcnn-wav2lip/0af8581d4a1842a8'

def make_batch():
    window = []

    for filename in os.listdir(basedir)[:syncnet_T]:
        path = f'{basedir}/{filename}'
        img = cv_loader(path)
        # img = np.expand_dims(img, axis=0)
        window.append(img)

    print(window[0].shape)
    x = np.concatenate(window, axis=2) / 255.
    print('START', x.shape)
    x = x.transpose(2, 0, 1)
    print(x.shape)

    torch_x = torch.FloatTensor(x)
    torch_x = torch.unsqueeze(torch_x, 0)
    print('TT', torch_x.shape)
    return torch_x


torch_batch = torch.cat([make_batch() for k in range(5)])
print('INPUT SHAPE', torch_batch.shape)
face_embeds = model.face_encoder(torch_batch)
print('FACE SHAPE', face_embeds.shape)

frame = torch_batch[0][0]
plt.imshow(frame, interpolation='nearest')
plt.show()