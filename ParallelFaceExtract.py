try:
    from DeepfakeDetection.FaceExtractor import FaceExtractor
except ModuleNotFoundError:
    from .DeepfakeDetection.FaceExtractor import FaceExtractor

import multiprocessing as mp
import queue as base_queue
import time
import cv2

from loader import load_video
from multiprocessing import Process, Manager, Queue
from itertools import repeat

def kwargify(**kwargs):
    return kwargs

class ParallelFaceExtract(object):
    def __init__(self, filepaths=None):
        self.filepaths = filepaths
        self.filepath_queue = Queue()

        self.manager = None
        self.extractions = None
        self.num_processes = None
        self.processes = None
        self.size = None

    @property
    def is_empty(self):
        if self.extractions is None:
            return True

        return len(self.extractions) == 0

    @property
    def is_done(self):
        return self.size == 0

    def pop(self):
        try:
            filepath = list(self.extractions.keys())[0]
        except IndexError:
            return None

        face_image_map = self.extractions[filepath]
        del self.extractions[filepath]
        self.size -= 1

        if self.size == 0:
            print(f'KILLING PROCESSES')
            for k in range(self.num_processes):
                self.filepath_queue.put('KILL')

            while self.filepath_queue.qsize() > 0:
                time.sleep(0.1)

            time.sleep(1)
            print(f'PROCESSES KILLED')

        return filepath, face_image_map

    def start(
        self, filepaths=None, base_dir=None,
        num_processes=6, max_cache_size=16, verbose=False,
        every_n_frames=20
    ):
        if filepaths is None:
            filepaths = self.filepaths
        if base_dir is not None:
            if base_dir.endswith('/'):
                base_dir = base_dir[:-1]

        assert filepaths is not None
        self.num_processes = num_processes
        self.size = len(filepaths)

        processes = []
        manager = Manager()
        self.manager = manager
        self.extractions = manager.dict()

        for k in range(num_processes):
            kwargs = kwargify(
                process_no=k, input_queue=self.filepath_queue,
                shared_dict=self.extractions,
                max_cache_size=max_cache_size, verbose=verbose,
                every_n_frames=every_n_frames
            )
            process = Process(
                target=self.process_filepaths, kwargs=kwargs
            )

            processes.append(process)
            process.daemon = True
            process.start()

        self.processes = processes

        for filepath in filepaths:
            if base_dir is not None:
                filepath = f'{base_dir}/{filepath}'

            self.filepath_queue.put(filepath)
            if verbose:
                print(self.filepath_queue.qsize())

    @staticmethod
    def process_filepaths(
        process_no, input_queue, shared_dict, max_cache_size,
        verbose=False, every_n_frames=20
    ):
        while True:
            if len(shared_dict) > max_cache_size:
                time.sleep(0.2)
                continue

            try:
                filepath = input_queue.get_nowait()
            except base_queue.Empty:
                if verbose:
                    size = input_queue.qsize()
                    print(f'{process_no} waiting [{size}]')

                time.sleep(1)
                continue

            if filepath == 'KILL':
                break

            # print(f'READING FILEPATH {filepath}')
            video_cap = cv2.VideoCapture(filepath)
            width_in = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_in = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            scale = 0.5
            if min(width_in, height_in) < 700:
                scale = 1

            print(f'{filepath} SCALE {scale}')
            vid_obj = load_video(
                video_cap, every_n_frames=every_n_frames,
                scale=scale
            )

            vid_obj = vid_obj.auto_resize()
            np_frames = vid_obj.out_video
            face_image_map = FaceExtractor.faces_from_video(
                np_frames, rescale=1, filename=filepath,
                every_n_frames=every_n_frames,
                coords_scale=scale
            )

            if verbose:
                print(f'PROCESSED [{process_no}] {filepath}')

            shared_dict[filepath] = face_image_map
