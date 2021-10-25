import cv2

from PIL import Image, ImageDraw
from loader import load_video

# 0e989500ec1080cf.mp4
# 0c0c3a74ba96c692.mp4
# bb34433231a222e5.mp4
# 0e5769008d488797.mp4

filename = '0e5769008d488797.mp4'
filepath = f'datasets/train/videos/{filename}'

video_cap = cv2.VideoCapture(filepath)
width_in = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_in = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

scale = 0.5
if min(width_in, height_in) < 700:
    scale = 1

print(f'{filepath} SCALE {scale}')
vid_obj = load_video(
    video_cap, every_n_frames=20,
    scale=scale
)

blackout = vid_obj.cut_blackout2(vid_obj.out_video)
left, right, top, bottom = blackout.to_tuple()
np_frames = vid_obj.out_video
image = np_frames[0]

pil_img = Image.fromarray(image)
img_draw = ImageDraw.Draw(pil_img)
shape = [(left, top), (right, bottom)]
img_draw.rectangle(shape, outline='#AAFF00', width=10)
pil_img.show()


