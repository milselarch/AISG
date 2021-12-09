import os
import numpy as np
import cv2
import random

from torchvision import transforms
from PIL import Image, ImageOps
from matplotlib import pyplot as plt

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

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

    cv_img = cv2.imread(path)
    pil_img = pil_loader(path)
    pil_np_img = np.array(pil_img)
    assert pil_np_img.shape == cv_img.shape

    convert = transform(pil_img)
    convert = convert[:, convert.shape[1]//2:]

    img = convert.numpy()
    print(convert.shape)
    img = img.transpose(1, 2, 0)

    img = np.array(img)
    # img = transform(img)
    window.append(img)

print(window[0].shape, len(window))
x = np.concatenate(window, axis=2) / 255.
print(x.shape)

x = x.transpose(2, 0, 1)
# x = x[:, x.shape[1] // 2:]
print(x.shape)

frame = x[1]
plt.imshow(frame, interpolation='nearest')
plt.show()