from FaceExtractor import FaceExtractor

extractor = FaceExtractor(scale_down=1)
extractor.extract_faces(
    filenames=['ab56e7569516c8ff.mp4'],
    export_dir='faces', pre_resize=False
)

extractor.extract_faces(pre_resize=False)
