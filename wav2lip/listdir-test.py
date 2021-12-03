import os
import time
import glob

basedir = '../datasets/extract/mtcnn-lip/0a3e16d70c766db6'

def run(turns=10000):
    start_time = time.perf_counter()
    for k in range(turns):
        # filenames = os.listdir(basedir)
        filenames = os.scandir(basedir)
        # filenames = list(filter(lambda x: x.name, filenames))
        filenames = [x.name for x in filenames]

    end_time = time.perf_counter()
    duration = end_time - start_time
    time_per_iter = duration / turns

    print(f'time taken {duration}')
    print(f'time per loop: {time_per_iter}')


run()
# iterator = list(os.scandir(basedir))
# print([x.name for x in iterator])