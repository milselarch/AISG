import numpy as np

import datasets
import pandas as pd
import time
import cv2

def horizontal_clip(image, intervals=5, roll=5):
    first_indexes, last_indexes = [], []
    interval_length = len(image) // intervals

    for k in range(intervals):
        interval = interval_length * k
        h_strip = image[interval]
        pool_strip = pd.DataFrame(h_strip).rolling(roll).median()
        pool_strip = pool_strip.to_numpy().ravel()
        clip_strip = pool_strip[roll-1:]
        # print('CLIP STRIP', clip_strip)

        indexes = np.where(clip_strip != 0)[0]
        last_normal_index = max(indexes)
        # print(indexes, last_normal_index)
        last_normal_index += roll - 1

        if pool_strip[roll] == 0:
            indexes = np.where(clip_strip != 0)[0]
            first_normal_index = min(indexes)
            # print(indexes, first_normal_index)
            first_normal_index += roll // 2 - 1
        else:
            first_normal_index = 0

        first_indexes.append(first_normal_index)
        last_indexes.append(last_normal_index)

    clip_start = int(np.median(first_indexes))
    clip_end = int(np.median(last_indexes))
    return clip_start, clip_end


dataset = datasets.Dataset()
filename = 'b57a96e8a6eb5db7.mp4'
video = dataset.load_video(filename)
cv2.namedWindow('display', cv2.WINDOW_NORMAL)
# cv2.namedWindow("frame", 0)
# cv2.resizeWindow("frame", 300, 300)

print(video)

ret, frame = video.read()
height, width = frame.shape[:2]

frame = cv2.resize(frame, (width // 4, height // 4))
height, width = frame.shape[:2]

img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_, binary_img = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
blackout_start, blackout_end = horizontal_clip(binary_img)
p_blackout = (blackout_end - blackout_start) / width
print(f'% BLACKOUT: {p_blackout}')

print(binary_img)
print(blackout_start, blackout_end)

# cv2.imshow('img_gray', binary_img)
video.set(cv2.CAP_PROP_POS_FRAMES, -1)

while video.isOpened():
    ret, frame = video.read()
    print(ret)

    if ret:
        new_frame = frame[:, blackout_start * 4: blackout_end * 4]
        new_frame = cv2.resize(new_frame, (width, height))
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