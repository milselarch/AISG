import multiprocessing as mp
import queue as base_queue
import time
import cv2

from loader import load_video
from DeepfakeDetection.FaceExtractor import FaceExtractor
from multiprocessing import Process, Manager, Queue
from itertools import repeat

class ParallelFaceExtract(object):
    def __init__(self, filepaths=None):
        self.filepaths = filepaths
        self.filepath_queue = Queue()
        self.extractions = None

    def start(
        self, filepaths=None, num_processes=6,
        max_cache_size=16, verbose=False
    ):
        if filepaths is None:
            filepaths = self.filepaths

        assert filepaths is not None

        processes = []
        manager = Manager()
        self.extractions = manager.dict()

        for k in range(num_processes):
            process = Process(
                target=self.process_filepaths, args=(
                    k, self.filepath_queue, self.extractions,
                    max_cache_size, verbose
                )
            )

            processes.append(process)
            process.daemon = True
            process.start()

        for filepath in filepaths:
            self.filepath_queue.put(filepath)
            if verbose:
                print(self.filepath_queue.qsize())

    @staticmethod
    def process_filepaths(
        process_no, input_queue, shared_dict, max_cache_size,
        verbose=False
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

            video_cap = cv2.VideoCapture(filepath)
            width_in = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_in = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            scale = 0.5
            if min(width_in, height_in) < 700:
                scale = 1

            vid_obj = load_video(
                video_cap, every_n_frames=20, scale=scale
            )

            vid_obj = vid_obj.auto_resize()
            np_frames = vid_obj.out_video
            face_image_map = FaceExtractor.faces_from_video(
                np_frames, rescale=1, filename=filepath
            )

            if verbose:
                print(f'PROCESSED [{process_no}] {filepath}')

            shared_dict[filepath] = face_image_map
