from data import Data
from pca import PCAPlotting
from kmeans import Kmeans
import symptomsPopularity

dataset = Data()
dataset.fill_na(0)
dataset.filter_out_zeros()
pca_data = PCAPlotting(dataset.d)
pca_data.reduce_dimensionality(2)
#pca_data.add_hospitalized_new()
kmeans = Kmeans(pca_data.reduced_data)
kmeans.find_clusters(8)
kmeans.plot_kmeans()

dataset.filter_out_zeros()
dataset.keep_15_symptoms()
dataset.merge_regions()
symp_pop = symptomsPopularity.SymptomPopularity(dataset.d)
symp_pop.symptoms_popularity()
