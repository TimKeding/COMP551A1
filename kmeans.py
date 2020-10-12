import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
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

    def plot_3d_kmeans(self):
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.data[:, 0], self.data[:, 1], self.data[:, 2], c=self.labels,
                     linewidth=0, antialiased=False)

        plt.show()
        return

    def intersect(self, a, b):
        """ return the intersection of two lists """
        return list(set(a) & set(b))

    def consistency(self, labels_high_dim, labels_low_dim):
        list1 = [[], [], [], [], [], [], [], []]
        for i in range(len(labels_high_dim)):
            list1[labels_high_dim[i]].append(i)

        list2 = [[], [], [], [], [], [], [], []]
        for i in range(len(labels_low_dim)):
            list2[labels_low_dim[i]].append(i)

        intersec = []
        for j in range(len(list1)):
            for k in range(len(list2)):
                intersec.append((len(self.intersect(list1[j], list2[k])), (j, k)))
        percentage_intersec = []
        for i in range(8):
            maximum = max(intersec)
            coord = maximum[1]
            mean_lists_length = (len(list1[coord[0]]) + len(list2[coord[1]])) / 2
            percentage_intersec.append((maximum[0] / mean_lists_length) * 100)
            intersec.remove(max(intersec))
        return (percentage_intersec, sum(percentage_intersec) / len(percentage_intersec))


