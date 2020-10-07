from data import Data
from pca import PCAPlotting

dataset = Data()
trimmed_dataset = dataset.keep_15_symptoms()
pca_data = PCAPlotting(trimmed_dataset)
pca_data.reduce_dimensionality(2)
pca_data.plot_data()
