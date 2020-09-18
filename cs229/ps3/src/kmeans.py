import numpy as np
import cv2

from sklearn.cluster import KMeans

if __name__ == "__main__":
    A = cv2.imread('../data/peppers-large.tiff')
    m, n, c = A.shape
    samples = A.reshape(-1, 3)
    kmeans = KMeans(n_clusters=16, random_state=0)
    res = kmeans.fit_predict(samples)
    A_compressed = kmeans.cluster_centers_[res].reshape((m, n, c)).astype(np.uint8)
    cv2.imshow('pic', A_compressed)
    cv2.waitKey(0)
