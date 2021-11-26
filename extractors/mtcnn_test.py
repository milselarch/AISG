import os
import torch

from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=50,
    thresholds=[0.6, 0.7, 0.7], factor=0.709,
    post_process=True, device=device
)

radius = 10.0
folder = '../datasets/extract/mtcnn-sync/0a0c9b2eeb2fd748'
# left eye, right eye, nose, left lip, right lip

for filename in os.listdir(folder):
    image_path = f'{folder}/{filename}'
    image = Image.open(image_path)
    bboxes, bconfs, landmarks = mtcnn.detect(
        image, landmarks=True
    )

    print(bboxes, bconfs, landmarks)
    draw = ImageDraw.Draw(image)
    landmark = landmarks[0]

    for k, point in enumerate(landmark[-2:]):
        print('POINT', point)
        x, y = [int(val) for val in point]
        sides = k + 3

        draw.regular_polygon(
            (x, y, radius), n_sides=sides, fill="red"
        )

    image.show()
    input('>>> ')
