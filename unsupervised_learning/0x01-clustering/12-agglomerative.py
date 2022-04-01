#!/usr/bin/env python3
"""
    Functions:
        def agglomerative(X, dist)
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """ expectation maximization
        Args:
            X: is a numpy.ndarray of shape (n, d) containing the dataset
            dist: is the maximum cophenetic distance for all clusters
            Performs agglomerative clustering with Ward linkage
            Displays the dendrogram with each cluster displayed in a
                different color
        Returns:
        clss, a numpy.ndarray of shape (n,) containing the cluster
            indices for each data point
    """
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    fig = plt.figure(figsize=(15, 8))
    clust = scipy.cluster.hierarchy.fcluster(linkage, dist, criterion='distance')
    dn = scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()
    return clust
