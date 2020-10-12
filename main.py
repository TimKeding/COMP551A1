from data import Data
from pca import PCAPlotting
from kmeans import Kmeans
import symptomsPopularity
import numpy as np

dataset = Data()
dataset.fill_na(0)
dataset.filter_out_zeros()

symptoms = dataset.d.columns[8:-1]
symptom_data = dataset.d.loc[:,symptoms].values
kmeans0 = Kmeans(symptom_data)
labels_high_dim = kmeans0.find_clusters(8)

pca_data = PCAPlotting(dataset.d)
pca_data.reduce_dimensionality(3)
#pca_data.add_hospitalized_new()
kmeans = Kmeans(pca_data.reduced_data)
labels_low_dim = kmeans.find_clusters(8)

consistency = kmeans.consistency(labels_high_dim, labels_low_dim)
print("{ Consistency percentages = ", consistency[0], '\n',
      "Percentage of points falling in the same cluster together again = ",
      consistency[1], " }")

kmeans.plot_kmeans()


dataset.filter_out_zeros()
dataset.keep_15_symptoms()
dataset.merge_regions()
symp_pop = symptomsPopularity.SymptomPopularity(dataset.d)
symp_pop.symptoms_popularity()

