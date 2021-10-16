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

clusterizer = FaceCluster()
# clusterizer.manual_grade_cross()
# clusterizer.sub_grade_clusters()
# clusterizer.auto_fill_fake_clusters()
# clusterizer.fill_mixed_clusters(False)
# clusterizer.analyse_distances()
clusterizer.manual_label_mixed()

print('DONE')