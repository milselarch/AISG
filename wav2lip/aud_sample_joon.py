import audio
import torch
import python_speech_features
import numpy as np

from scipy.io import wavfile
from hparams import hparams, get_image_list
from models.syncnet import SyncNet_color
from models.syncnet_joon import SyncnetJoon

model = SyncnetJoon()
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

print('INPUT SHAPE', cc_in.shape)
audio_embeds = model.forward_aud(cc_in)
print('AUDIO SHAPE', audio_embeds.shape)
