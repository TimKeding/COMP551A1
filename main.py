from data import Data
from kmeans import Kmeans
from pca import PCAPlotting
import symptomsPopularity

dataset = Data()
dataset.keep_15_symptoms()
dataset.fill_na(0)
pca_data = PCAPlotting(dataset.d)
pca_data.reduce_dimensionality(3)
kmeans = Kmeans(pca_data.reduced_data)
kmeans.find_clusters(5)
kmeans.plot_kmeans()
dataset.merge_regions()

sym_pop = symptomsPopularity.SymptomPopularity(dataset.d)
sym_pop.symptoms_popularity()



