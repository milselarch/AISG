from FaceAnalysis import FaceCluster

"""
confirm fakes: 1438
classified fakes: 661
fakes: 2099

classified reals: 559
labelled reals: 859
reals: 1418

total: 3517

1112 fakes in mixed df
4380 fakes total (all sources)
"""

"""
total 6943
total fakes 4380
swap fakes 707
both fakes 1292
audio only fakes 697
face only fakes 1682
total reals 2563
"""

clusterizer = FaceCluster()
# clusterizer.manual_grade_cross()
# clusterizer.sub_grade_clusters()
# clusterizer.auto_fill_fake_clusters()
# clusterizer.fill_mixed_clusters(False)
# clusterizer.analyse_distances()
# clusterizer.manual_label_mixed()
# clusterizer.stitch_labels()
# clusterizer.generate_all_labels(False)
# clusterizer.get_swap_videos()
# clusterizer.count_video_faces()
# clusterizer.manual_label_two_face()
# clusterizer.face_name_shift()
# clusterizer.verify_detections()
clusterizer.detections_label()

print('DONE')