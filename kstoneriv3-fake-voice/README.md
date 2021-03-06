# Fake-Voice-Detection

Author: Jingqiu Ding, Kei Ishikawa, Xiaoran Chen. 
The original code for [Cyclic GAN](https://github.com/leimao/Voice_Converter_CycleGAN) is by Lei Mao.<br>

Environment: ubuntu 18.04, Python 3.6

## Converted Voice Sample
|     id | real                                                                             | fake                                                                             |
|-------:|:---------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|
| 200001 | link: [./converted_samples/200001_real.wav](./converted_samples/200001_real.wav) | link: [./converted_samples/200001_fake.wav](./converted_samples/200001_fake.wav) |
| 200002 | link: [./converted_samples/200002_real.wav](./converted_samples/200002_real.wav) | link: [./converted_samples/200002_fake.wav](./converted_samples/200002_fake.wav) |
| 200003 | link: [./converted_samples/200003_real.wav](./converted_samples/200003_real.wav) | link: [./converted_samples/200003_fake.wav](./converted_samples/200003_fake.wav) |
| 200004 | link: [./converted_samples/200004_real.wav](./converted_samples/200004_real.wav) | link: [./converted_samples/200004_fake.wav](./converted_samples/200004_fake.wav) |
| 200005 | link: [./converted_samples/200005_real.wav](./converted_samples/200005_real.wav) | link: [./converted_samples/200005_fake.wav](./converted_samples/200005_fake.wav) |

\* depending on the browser, you cannot play the wav files on the browser. 

## plot of Score* for GMM-based verification system
\* "score" is the log likelihood ratio of the GMM-Speakermodel and the GMM-UBG model.
![](<./out/plot/average_score_per_small_clip_for_test,fake,ubg.svg>)
- test ... Obama's voice which is not used for training neither conversion system or verificationsystem
- fake ... fake voice of Obama generated by voice conversion system (cycle GAN)
- universal background ... voice from a lot of people


## (FOR LEONHARD CLUSTER)

run the following at `.../Fake-Voice-Detection/`
```bash
source ./set_env_leonhard.sh
bsub -W 4:00 -R "rusage[ngpus_excl_p=1,mem=16000]" source ./run_all_leonhard.sh
```

## Introduction


## Files

```
.
?????????src
???   ?????????conversion
???   ???   ?????? model.py
???   ???   ?????? module.py
???   ???   ?????? preprocess.py
???   ???   ?????? train.py
???   ???   ?????? utils.py
???   ?????????verification_gmm
???   ???   ?????? compute_auc.py
???   ???   ?????? train_and_plot.py
???   ????????? verification_vae
???   ???   ?????? cvae_verification.py
???   ???   ?????? cvae_keras.py
???   ????????? download.py
???   ????????? split_normalize_raw_speech.py.py
???
?????????data
???   ?????????target_raw (Obama)
???   ?????????target (Obama)
???   ???   ?????? train_conversion
???   ???   ?????? train_verification
???   ???   ?????? test
???   ?????????source
???   ???   ?????? train_conversion
???   ?????????ubg
???       ?????? train_verification
???       ?????? test
?????????out
???   ?????????plot
???   ?????????scores
????????? set_env_leonhard.sh
????????? run_all_leonhard.sh
????????? README.md
```


## Requirments
Install all the requirements (except numpy, matplotlib, scikitlearn, tensorflow).

```bash
pip install --user -r requirements.txt
```
If librosa gives backend error, run following. (This is `module load ffmpeg` in HPC cluster in ETH.)
```bash
apt-get install ffmpeg
```

## Usage
run the following at `.../Fake-Voice-Detection/`

### Download Dataset and preprocess
Download and unzip datasets and pretrained models.

```bash
$ python ./src/download.py
```

Split the raw speech
```bash
$ python ./src/split_normalize_raw_speech.py
```

### CycleGAN Voice Conversion
Train the Voice Conversion Model
```bash
$ python ./src/conversion/train.py --model_dir='./model/conversion/pretrained'
```

Convert the source speaker's voice
```bash
$ python ./src/conversion/convert.py --model_dir='./model/conversion/pretrained'
```

### GMM-UBG verification
Train the GMM based verification system and Plot the scores
```bash
$ python ./src/verification_gmm/train_and_plot.py
```

compute AUC converted samples of every 50 epoch
```bash
$ python ./src/verification_gmm/compute_auc.py
```


### Convolutional VAE

Train the CVAE based verification system and Plot the scores

```bash
$ python ./src/verification_cvae/cvae_verification.py
```


