from FaceAnalysis import FaceCluster

clusterizer = FaceCluster()
files = [
    '1db09b637dabdce6.mp4',
    'f5bec754e106e26a.mp4',
    'c7991f7928d79ffc.mp4',
    'f6118def8c88902d.mp4',
    '9a22372d22a52397.mp4',
    'fe17857f491c29d2.mp4'
]

result = clusterizer.make_background_clusters(files)
clusters, distances, background_hashes, face_hashes = result
print(clusters, distances)

clusterizer.cluster()
# clusterizer.get_clusters_info()

print('DONE')