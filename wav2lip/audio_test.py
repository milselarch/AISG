import audio
import torch

from hparams import hparams, get_image_list
from models.syncnet_v2 import SyncNet_color

model = SyncNet_color()
syncnet_mel_step_size = 16

audio_path = f'../datasets-local/audios-flac/aa60f634c1594678.flac'
wav = audio.load_wav(audio_path, hparams.sample_rate)
orig_mel = audio.melspectrogram(wav).T
mels = []

# length = len(orig_mel)
length = 5

for k in range(length):
    mel = orig_mel[0:syncnet_mel_step_size, :]
    torch_mel = torch.FloatTensor(mel.T).unsqueeze(0).unsqueeze(0)
    mels.append(torch_mel)
    # assert mel.shape[0] == syncnet_mel_step_size

melp = torch.cat(mels)
print('INPUT SHAPE', melp.shape)
audio_embeds = model.audio_encoder(melp)
print('AUDIO SHAPE', audio_embeds.shape)
