from FaceExtractor import FaceExtractor

# [labels[labels['filename'] == f]['label'] for f in mismatches]
# [labels[labels['filename'] == f]['label'].to_numpy()[0] for f in mismatches]
# mismatches = np_labels[~np.in1d(np_labels, filenames)]
# a50ed6791d19116b
# 61ba23efc5776bf5
# 8ff5133f687d20d4.mp4
# 8f8f2be1766a03d4.mp4
# bb34433231a222e5

extractor = FaceExtractor(scale_down=1)
extractor.extract_faces(
    filenames=['bb34433231a222e5.mp4'],
    export_dir='faces', pre_resize=False
)

extractor.extract_faces(pre_resize=False)
