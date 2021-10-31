import os
import numpy as np
import cv2
import random

import torch
from torchvision import transforms
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from models.syncnet import SyncNet_color

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

model = SyncNet_color(107)

def pil_loader(path: str, mirror_prob=0.5) -> Image.Image:
    with open(path, 'rb') as f:
        image = Image.open(f)
        if random.random() < mirror_prob:
            image = ImageOps.mirror(image)

        return image.convert('RGB')


basedir = '../datasets-local/mtcnn-faces/3a3d68bceddb6dab'
window = []

for filename in os.listdir(basedir):
    path = f'{basedir}/{filename}'

    pil_img = pil_loader(path)
    pil_np_img = np.array(pil_img)
    # assert pil_np_img.shape == cv_img.shape

    # img = np.array(img)
    # img = transform(img)
    window.append(pil_np_img)

print(window[0].shape, len(window))

x = np.concatenate(window, axis=2) / 255.
print(x.shape)
x = x.transpose(2, 0, 1)
x = x[:, x.shape[1] // 2:]
print(x.shape)

torch_x = torch.FloatTensor(x)
torch_x = torch.unsqueeze(torch_x, 0)
face_embeds = model.face_encoder(torch_x)

# print(face_embeds)
print(face_embeds.shape)

frame = x[1]
plt.imshow(frame, interpolation='nearest')
plt.show()