import numpy as np
import datasets
import pandas as pd
import time
import cv2

from loader import VideoArray

dataset = datasets.Dataset()
# filename = 'b57a96e8a6eb5db7.mp4'
# filename = 'e7c1248f1566506d.mp4'
filename = '4d1c4b1e7f86f6bf.mp4'
video = dataset.load_video(filename)
cv2.namedWindow('display', cv2.WINDOW_NORMAL)
# cv2.namedWindow("frame", 0)
# cv2.resizeWindow("frame", 300, 300)

print(video)

ret, frame = video.read()
height, width = frame.shape[:2]

downscale = 1
frame = cv2.resize(frame, (width // downscale, height // downscale))
height, width = frame.shape[:2]

img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_, binary_img = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
blackout_h_start, blackout_h_end = VideoArray.h_clip(binary_img)
blackout_v_start, blackout_v_end = VideoArray.v_clip(binary_img)
ph_blackout = (blackout_h_end - blackout_h_start) / width
pv_blackout = (blackout_v_end - blackout_v_start) / height
print(f'H % BLACKOUT: {ph_blackout}')
print(f'V % BLACKOUT: {pv_blackout}')

print(binary_img)
# print(blackout_start, blackout_end)

# cv2.imshow('img_gray', binary_img)
video.set(cv2.CAP_PROP_POS_FRAMES, -1)

while video.isOpened():
    ret, frame = video.read()
    # print(ret)

    if ret:
        new_frame = frame[
            blackout_v_start * downscale: blackout_v_end * downscale,
            blackout_h_start * downscale: blackout_h_end * downscale
        ]

        new_frame = cv2.resize(new_frame, (width, height))
        # new_frame = frame
        cv2.imshow('display', new_frame)
    else:
        break

    # print(frame)
    if cv2.waitKey(1) == 27:
        break  # esc to quit

    time.sleep(1 / 30)
    # input('test >>> ')

# video.release()
# cv2.destroyAllWindows()