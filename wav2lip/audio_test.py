import audio
import torch

from hparams import hparams, get_image_list
from models.syncnet import SyncNet_color

model = SyncNet_color()
syncnet_mel_step_size = 16

audio_path = f'../datasets-local/audios-flac/aa60f634c1594678.flac'
wav = audio.load_wav(audio_path, hparams.sample_rate)
orig_mel = audio.melspectrogram(wav).T
mels = []


for k in range(len(orig_mel)):
    mel = orig_mel[0:syncnet_mel_step_size, :]
    mels.append(mel.T)
    assert mel.shape[0] == syncnet_mel_step_size

melp = torch.FloatTensor(mels).unsqueeze(0)
audio_embeds = model.audio_encoder(melp)
print(audio_embeds)
