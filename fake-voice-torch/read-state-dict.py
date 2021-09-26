import pandas as pd
import numpy as np
import os

from trainer import Trainer
from tqdm.auto import tqdm


trainer = Trainer(cache_threshold=20, load_dataset=False)
state_dict = trainer.model.state_dict()
print(f'STATE DICT', state_dict)
print(len(state_dict))