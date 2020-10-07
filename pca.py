import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm


class PCAPlotting:

    def __init__(self, data: pd.DataFrame):
        self.plot_dimensions = 2
        self.data = data
        self.reduced_data = None

    def reduce_dimensionality(self, new_dimension=2):
        self.plot_dimensions = new_dimension
        # strip out symptoms from merged dataset
        symptoms = self.data.columns[8:-1]
        symptom_data = self.data.loc[:, symptoms].values
        symptom_data = StandardScaler().fit_transform(symptom_data)

        pca = PCA(n_components=new_dimension)
        self.reduced_data = pca.fit_transform(symptom_data)

        return

    def plot_data(self, surface=False):
        if self.plot_dimensions == 2:
            self.plot_2d_pca()
        else:
            if surface:
                self.plot_3d_surface_pca()
            else:
                self.plot_3d_points_pca()
        return

    def plot_2d_pca(self):
        plt.scatter(self.reduced_data[:, 0], self.reduced_data[:, 1],
                    linewidth=0, antialiased=False)

        plt.axis('equal')
        plt.show()
        return

    def plot_3d_surface_pca(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        surf = ax.plot_trisurf(self.reduced_data[:, 0], self.reduced_data[:, 1], self.reduced_data[:, 2], cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        return

    def plot_3d_points_pca(self):
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.reduced_data[:, 0], self.reduced_data[:, 1], self.reduced_data[:, 2],
                     linewidth=0, antialiased=False)

        plt.show()
        return
