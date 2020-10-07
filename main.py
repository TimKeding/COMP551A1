from data import Data
from pca import PCAPlotting
from kmeans import Kmeans

dataset = Data()
pca_data = PCAPlotting(dataset.data)
pca_data.reduce_dimensionality(3)
kmeans = Kmeans(pca_data.reduced_data)
kmeans.find_clusters(8)
kmeans.plot_kmeans()
