import os
import utils
import numpy as np

from tqdm.auto import tqdm

cache = {}


def resolve(filename, start=160, clip=80, cache_mel=True):
    try:
        name = filename[:filename.index('.')]
    except ValueError as e:
        print('bad filename', filename)
        raise e

    file_path = f'../datasets-local/audios-flac/{name}.flac'

    if not os.path.exists(file_path):
        invalid.append(file_path)
        raise FileNotFoundError

    if name in cache:
        return cache[name]

    mel, _, duration = utils.process(file_path, '', 0)
    mel = mel[start: start + clip]

    if cache_mel:
        cache[name] = mel

    return mel


fakes = [
    'dad726f28632a03d.mp4',
    '9c98011528f2b9c9.mp4',
    '4c872bdbe7b17f10.mp4',
    'e42bde7c614b229a.mp4',
    '9c6dcef6570e674d.mp4',
    '433c364a65ddd802.mp4',
    '4e05a08ee5d1ed9a.mp4',
    '2b94598877187c2c.mp4'
]

fakes = [
    'ac6bdabe440a83d8.mp4',
    'd0d53e4b58a7e3e9.mp4',
    '6c8e3350fa5fc133.mp4',
    '04969e096070508e.mp4',
    'bb80ef2d536f4c65.mp4',
    'b3512c48bf8083c9.mp4',
    '3431bc0a013a60e8.mp4',
    '00a35da6423560a2.mp4',
    '745d298e17242203.mp4',
    'a1f0dffc4482f91c.mp4'
]

"""
fakes = [
   'd0d53e4b58a7e3e9.mp4',
   '3431bc0a013a60e8.mp4',
   '00a35da6423560a2.mp4',
   '745d298e17242203.mp4',
   'a1f0dffc4482f91c.mp4'
]
"""

fakes = [
    '8299db678c0be97c.mp4',
    'a7a18e6267d2e6bb.mp4',
    'd792f12b1844af72.mp4',
    '2566f4625f3cb0e3.mp4',
    '549ac57bd72ef965.mp4',
    'b076ef1ece9347a5.mp4',
    '37b94333fdd0942c.mp4',
    'f9313bbd24a94d22.mp4',
    'f0b430a807005be3.mp4',
    '4b816ede22e6544c.mp4'
]

fakes = [
    '4b59937e35c3eb7d.mp4',
    'b41fcbdd0f0493fd.mp4',
    '777ce549e33e1cc2.mp4',
    'a67a35867b48fce3.mp4',
    'c2161c8509b5ff6f.mp4'
]

fakes = """
c7991f7928d79ffc.mp4
f6118def8c88902d.mp4
d3fd00209a14becd.mp4
d6db9870503ea1dc.mp4
027389ba72f41bcb.mp4
f1b68d2cca9d9edf.mp4
4a04051d3c7f7da4.mp4
4a19ad2138d7f53d.mp4
028b889d804be45c.mp4
f5bec754e106e26a.mp4
""".strip().split('\n')

distances = []
clusters = []

pbar = tqdm(range(len(fakes)))

while len(fakes) > 0:
    new_fakes = []
    base_filename = fakes[0]
    base_mel = resolve(base_filename)
    cluster_dist = [0]
    cluster = [base_filename]
    pbar.update()

    for filename in fakes[1:]:
        mel = resolve(filename)
        distance = np.linalg.norm(mel - base_mel)
        cluster_dist.append(distance)

        if distance > 17:
            new_fakes.append(filename)
        else:
            cluster.append(filename)
            pbar.update()

    clusters.append(cluster)
    distances.append(cluster_dist)
    print(base_filename, fakes[1:])
    print(cluster_dist)

    fakes = new_fakes

print(distances)
print(clusters)
