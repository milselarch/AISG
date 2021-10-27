from FaceAnalysis import FaceCluster

clusterizer = FaceCluster()
# clusterizer.double_frames()
# clusterizer.fix_talkers()
# clusterizer.copy_talkers()
# clusterizer.get_face_ratios()
# clusterizer.purge_mismatches()
# clusterizer.stitch_mtcnn_detections()
# clusterizer.discern_mismatches()
# clusterizer.flip_detections()
clusterizer.manual_label_two_face()
# clusterizer.realign_coord_mtcnn()