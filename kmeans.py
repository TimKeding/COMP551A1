import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances_argmin, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class Kmeans:

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.dimensions = len(data[0])
        self.centers = None
        self.labels = None

    def find_clusters(self, num_clusters, rseed=2):

        rng = np.random.RandomState(rseed)
        i = rng.permutation(self.data.shape[0])[:num_clusters]
        self.centers = self.data[i]

        while True:
            self.labels = pairwise_distances_argmin(self.data, self.centers)
            new_centers = np.array([self.data[self.labels == i].mean(0)
                                    for i in range(num_clusters)])
            if np.all(self.centers == new_centers):
                break
            self.centers = new_centers
        return self.labels

    def plot_kmeans(self):
        if self.dimensions == 2:
            self.plot_2d_kmeans()
        else:
            self.plot_3d_kmeans()
        return

    def plot_2d_kmeans(self):
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels,
                    linewidth=0, antialiased=False)

        plt.axis('equal')
        plt.show()
        return

    def plot_3d_kmeans(self, view_init=None):
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.data[:, 0], self.data[:, 1], self.data[:, 2], c=self.labels,
                     linewidth=0, antialiased=False)
        if view_init:
            ax.view_init(view_init[0], view_init[1])
        plt.show()
        return

    def intersect(self, a, b):
        """ return the intersection of two lists """
        return list(set(a) & set(b))

    def consistency(self, num_clust, labels_high_dim, labels_low_dim):

        list1 = []
        list2 = []
        for i in range(num_clust):
            list1.append([])
            list2.append([])
        for i in range(len(labels_high_dim)):
            list1[labels_high_dim[i]].append(i)

        for i in range(len(labels_low_dim)):
            list2[labels_low_dim[i]].append(i)

        intersec = []

        for j in range(len(list1)):
            for k in range(len(list2)):
                intersec.append((len(self.intersect(list1[j], list2[k])), (j, k)))
        percentage_intersec = []
        for i in range(num_clust):
            if len(intersec) != 0:
                maximum = max(intersec)
                coord = maximum[1]
                mean_lists_length = (len(list1[coord[0]]) + len(list2[coord[1]])) / 2
                percentage_intersec.append((maximum[0] / mean_lists_length) * 100)
                intersec.remove(max(intersec))
        return (percentage_intersec, sum(percentage_intersec) / len(percentage_intersec))

def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        print(k)
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)

    return sse

def silhouette_method(x, kmax):
    sil = []
    for k in range(2, kmax + 1):
        print(k)
        kmeans = KMeans(n_clusters=k).fit(x)
        labels = kmeans.labels_
        sil.append(silhouette_score(x, labels, metric='euclidean'))
    return sil





