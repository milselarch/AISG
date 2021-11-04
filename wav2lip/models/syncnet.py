
import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2d

class SyncNet_color(nn.Module):
    def __init__(self, syncnet_T, num_dense_neurons=32):
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(
                syncnet_T * 3, 32, kernel_size=(7, 7),
                stride=1, padding=3
            ),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        )

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        )

        self.dense_layer = nn.ModuleList([
            nn.Linear(1024, num_dense_neurons),
            nn.Linear(num_dense_neurons, 1)
        ])

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
        return audio_embedding, face_embedding

    def predict(self, *args, **kwargs):
        audio_embed, face_embed = self(*args, **kwargs)
        # print('EMBEDS', audio_embed.shape, face_embed.shape)
        cosine_similarity = nn.functional.cosine_similarity
        similarity = cosine_similarity(audio_embed, face_embed)
        return 1.0 - similarity

    def pred_fcc(self, audios, images, fcc_ratio=0.8):
        audio_embeds, image_embeds = self(audios, images)
        cosine_similarity = nn.functional.cosine_similarity
        similarity = cosine_similarity(audio_embeds, face_embeds)
        cosine_fakeness = 1.0 - similarity

        flat_audio_embeds = torch.flatten(audio_embeds, start_dim=1)
        flat_image_embeds = torch.flatten(image_embeds, start_dim=1)
        flat_embeds = torch.cat(
            [flat_image_embeds, flat_audio_embeds], dim=-1
        )

        prediction = self.dense_layer(flat_embeds)
        combined_pred = (
            (1-fcc_ratio) * cosine_fakeness +
            fcc_ratio * prediction
        )

        return combined_pred