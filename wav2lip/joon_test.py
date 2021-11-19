import os
import numpy as np
import cv2
import random
import torch

from torchvision import transforms
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from models.syncnet_joon import SyncnetJoon
from hparams import hparams, get_image_list

transform = transforms.Compose([
    transforms.Resize((128, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

syncnet_T = 5
model = SyncnetJoon()

def pil_loader(path: str, mirror_prob=0.5) -> Image.Image:
    with open(path, 'rb') as f:
        image = Image.open(f)
        if random.random() < mirror_prob:
            image = ImageOps.mirror(image)

        return image.convert('RGB')

def cv_loader(
    img, mirror_prob=0.5, size=224,
    verbose=False
):
    if type(img) is str:
        img = cv2.imread(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if size is None:
        size = hparams.img_size

    assert img.shape[0] == img.shape[1]
    img = cv2.resize(img, (size, size))

    if random.random() < mirror_prob:
        # cls.m_print(verbose, 'FLIP')
        img = cv2.flip(img, 1)

    return img


# basedir = '../datasets-local/mtcnn-faces/3a3d68bceddb6dab'
basedir = '../datasets/extract/mtcnn-sync/1ddf59260ed2749f'

def make_batch():
    window = []

    for filename in os.listdir(basedir)[:syncnet_T]:
        path = f'{basedir}/{filename}'
        img = cv_loader(path)
        # img = np.expand_dims(img, axis=0)
        window.append(img)

    im = np.stack(window, axis=3)
    im = np.expand_dims(im, axis=0)
    im = np.transpose(im, (0, 3, 4, 1, 2))
    im_batch = torch.from_numpy(im.astype(float)).float()

    return im_batch


syncnet_mel_step_size = 20

audio_path = f'../datasets/extract/audios-flac/aa60f634c1594678.flac'
wav = audio.load_wav(audio_path, hparams.sample_rate)
orig_mel = audio.melspectrogram(wav).T
mels = []

# sample_rate, audio_mfcc = wavfile.read(audio_path)
mfcc = zip(*python_speech_features.mfcc(wav, hparams.sample_rate))
mfcc = np.stack([np.array(i) for i in mfcc])
cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

print('RAW AUDIO', wav.shape)
print(f'SAMPLE RATE', hparams.sample_rate)
print(f'MEL SHAPE', orig_mel.shape)
print(f'CCT SHAPE', cct.shape)
print(f'MFCC SHAPE', mfcc.shape)

# length = len(orig_mel)
cc_batch = [
    cct[:, :, :, vframe * 4: vframe * 4 + syncnet_mel_step_size]
    for vframe in range(10)
]
cc_in = torch.cat(cc_batch, 0)


torch_batch = torch.cat([make_batch() for k in range(5)])
print('INPUT SHAPE', torch_batch.shape)
face_embeds = model.forward_lip(torch_batch)
print('FACE EMBEDS SHAPE', face_embeds.shape)
flat_image_embeds = torch.flatten(face_embeds, start_dim=1)
print('FLAT FACE EMBEDS SHAPE', face_embeds.shape)

frame = torch_batch[0][0]
# plt.imshow(frame, interpolation='nearest')
# plt.show()