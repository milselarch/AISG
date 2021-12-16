from AudioAnalysis import AudioAnalysis

from tqdm.auto import tqdm

analyser = AudioAnalysis()
filenames = analyser.get_filenames()
cache, durations = {}, {}

for filename in tqdm(filenames):
    mel, _, duration = analyser.load_mel(filename)
    durations[filename] = duration
    cache[filename] = mel

print(f'{len(filenames)} FILES CACHED')
print('DONE')
input('>>> ')