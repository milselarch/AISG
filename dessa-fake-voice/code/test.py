import os
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# see issue #152
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Directory from which we read the data
mode = "unlabeled"  # real, fake, or unlabeled

# Convert files to flac
# convert_to_flac(os.path.join(data_dir,mode))

dirpath = '../../datasets/train/audios'
filenames = os.listdir(dirpath)
# filenames = [f'{dirpath}/{filename}' for filename in filenames]
filenames = filenames[:10]
# print(filenames)
# raise ValueError

# preprocess the files
processed_data = preprocess_from_filenames(
    filenames, dirpath, mode=None, use_parallel=False
)

print(f'PROCESSED DATA', processed_data[0])
print(len(processed_data[0][0]), processed_data[0][0].shape)

# Visualize the preprocessed data
plot_spectrogram(
    processed_data[0][0],
    path='visualize_inference_spectrogram.png'
)


# Load the pretrained model
pretrained_model_name = 'pretrained_model.h5'

discriminator = Discriminator_Model(
    load_pretrained=True,
    saved_model_name=pretrained_model_name
)

print("The probability of the clip being real is: {:.2%}".format(
    discriminator.predict_labels(
        processed_data[0], raw_prob=True, batch_size=20
    )[0][0]
))


