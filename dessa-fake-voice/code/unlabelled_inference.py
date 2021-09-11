import os
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# see issue #152
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Directory from which we read the data
data_dir = "../data/inference_data/"
mode = "unlabeled"  # real, fake, or unlabeled

# Convert files to flac
# convert_to_flac(os.path.join(data_dir,mode))


# preprocess the files
processed_data = preprocess_from_ray_parallel_inference(
    data_dir, mode, use_parallel=False
)

# Visualize the preprocessed data
plot_spectrogram(
    processed_data[0],
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
        processed_data, raw_prob=True, batch_size=20
    )[0][0]
))


