from FaceExtractor import FaceExtractor

# [labels[labels['filename'] == f]['label'] for f in mismatches]
# [labels[labels['filename'] == f]['label'].to_numpy()[0] for f in mismatches]
# mismatches = np_labels[~np.in1d(np_labels, filenames)]

extractor = FaceExtractor(scale_down=1)
extractor.extract_faces(
    filenames=['8475c38b094dee14.mp4'],
    export_dir='faces', pre_resize=False
)

# extractor.extract_faces(pre_resize=False)
