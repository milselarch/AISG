import pandas as pd
import numpy as np
import random as random
import torch
import os

from utils import hparams
from trainer import Trainer
from tqdm.auto import tqdm

trainer = Trainer(cache_threshold=20)
path = './saves/checkpoints/210928-1400/E35392_T0.86_V0.68.pt'
trainer.load_model(path)

def invoke(seed=420, randomize=False, validator=False):
    random.seed(seed)

    batch_x, np_labels = trainer.prepare_batch(
        batch_size=32, fake_p=0.5,
        target_lengths=(128, 1024),
        is_training=False
    )

    indices = np.arange(len(batch_x))

    if randomize:
        assert len(batch_x) == len(np_labels)
        random.shuffle(indices)
        batch_x = batch_x[indices]
        np_labels = np_labels[indices]
        print('RANDOMIZED')
        # print('RANDOMIZE', np_labels)

    labels = np.array([k.flatten()[0] for k in np_labels])
    torch_batch_x = torch.tensor(batch_x).to(trainer.device)
    torch_labels = torch.tensor(np_labels).float()
    torch_labels = torch_labels.to(trainer.device).detach()

    if not validator:
        preds = trainer.model(torch_batch_x)
    else:
        preds = trainer.batch_predict(torch_batch_x, to_numpy=False)

    o_preds = np.array([
        float(pred.flatten().mean()) for pred in preds
    ])

    print('PREDS', preds)
    print('INDICES', indices)
    print('LABELS', labels)

    print('O-PREDS', o_preds)

    trainer.optimizer.zero_grad()
    loss = trainer.criterion(preds, torch_labels)
    loss_value = loss.item()
    state = trainer.model.batch_norm_layers[0].state_dict()
    print('STATE DICT', state)
    print('LOSS VALUE', loss_value)
    print('END')


invoke()
invoke(validator=True)
# invoke(randomize=True)
# invoke(randomize=True)
print('DONE')