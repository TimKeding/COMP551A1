from data import Data
from pca import PCAPlotting
from kmeans import Kmeans

dataset = Data()
trimmed_dataset = dataset.keep_15_symptoms()
pca_data = PCAPlotting(trimmed_dataset)
pca_data.reduce_dimensionality(3)
kmeans = Kmeans(pca_data.reduced_data)
kmeans.find_clusters(5)
kmeans.plot_kmeans()
