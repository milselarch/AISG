import numpy as np
import imagehash
import pandas as pd
import cv2
import sys
import os

sys.path.append('')

from matplotlib import pyplot as plt
from loader import load_video
from PIL import Image

scale = 0.25
print(os.getcwd())
cwd = os.getcwd()
base_dir = f'datasets/train/videos'

# real = '8e23c73ba7e2848a.mp4'
# real = '4ba3229f76ac5cea.mp4'
# fake = '4ef85974dc0584ad.mp4'

# fake = 'f6118def8c88902d.mp4'
# real = '6100058cc8ffc0fd.mp4'
# fake = '63bed62257daccaf.mp4'
# real = '9a22372d22a52397.mp4'

real = '1b0d98d636ae20f1.mp4'
# fake = '9420e8ab38a84b09.mp4'
# fake = '9bc4f1306bb8e2cd.mp4'
fake = '0f9f831906a04e8d.mp4'

detect_path = 'stats/detections-20210926-112918.csv'
detect_df = pd.read_csv(detect_path)
cond = detect_df['filename'] == real
frame_data = detect_df[cond].iloc[0]
# buffer = 40

true_top = int(frame_data['top'])
true_left = int(frame_data['left'])
true_right = int(frame_data['right'])
true_bottom = int(frame_data['bottom'])
area = (true_bottom - true_top) * (true_right - true_left)
area_root = area ** 0.5
buffer = 0 * area_root // 7

print('AREA ROOT', area_root)
print('BUFFER', buffer)

top = max(int((frame_data['top'] - buffer) * scale), 0)
left = max(int((frame_data['left'] - buffer) * scale), 0)
right = max(int((frame_data['right'] + buffer) * scale), 0)
bottom = max(int((frame_data['bottom'] + buffer) * scale), 0)

fake_path = os.path.abspath(f'{base_dir}/{fake}')
real_path = os.path.abspath(f'{base_dir}/{real}')
print(fake_path)
print(real_path)

fake_video = cv2.VideoCapture(fake_path)
real_video = cv2.VideoCapture(real_path)
fake_video = load_video(fake_video, specific_frames=[20], scale=scale)
real_video = load_video(real_video, specific_frames=[20], scale=scale)

# print(fake_video.frame_mapping)
# print(real_video.frame_mapping)

def threshold(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 11
    )

def show(image, display=True):
    plt.imshow(image, interpolation='nearest')
    if display:
        plt.show()


image_fake = fake_video.out_video[0]
image_real = real_video.out_video[0]
thresh_fake = threshold(image_fake)
thresh_real = threshold(image_real)

t_hash1 = imagehash.phash(Image.fromarray(thresh_fake))
t_hash2 = imagehash.phash(Image.fromarray(thresh_real))

print('THRESHOLD HASHING')
print(t_hash1)
print(t_hash2)
print(t_hash1 == t_hash2)
print(t_hash1 - t_hash2)

show(thresh_fake, True)
show(thresh_real, True)
# plt.show()

hash1 = imagehash.phash(Image.fromarray(image_fake))
hash2 = imagehash.phash(Image.fromarray(image_real))

# show(image_fake)
# show(image_real)

print('NORMAL HASHING')
print(hash1)
print(hash2)
print(hash1 == hash2)
print(hash1 - hash2)
# print(f'DISTANCE {distance}')