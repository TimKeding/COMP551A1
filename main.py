from data import Data
from pca import PCAPlotting
from kmeans import *
import symptomsPopularity
import copy
from sklearn.preprocessing import StandardScaler

dataset = Data()
dataset.fill_na(0)
dataset.filter_out_zeros()
temp_dataset = copy.deepcopy(dataset)
symptoms = dataset.d.columns[8:-1]
symptom_data = dataset.d.loc[:,symptoms].values
symptom_data = StandardScaler().fit_transform(symptom_data)

#Commented out since it takes time to produce the graphs
#Finds the optimal number of clusters in full dimensions
#The same thing was done for 2D and 3D yielding the 3 ElbowResults graph
'''
x_vals = []
for i in range(1, 101):
    x_vals.append(i)
res = calculate_WSS(symptom_data, 100)
plt.plot(x_vals, res)
plt.xlabel("K")
plt.ylabel("WSS")
plt.suptitle("Elbow method results on symptoms search popularity in full dimensions")
plt.savefig("ElbowResInFullDims.jpg")
plt.show()


res = silhouette_method(symptom_data, 101)
plt.plot(x_vals, res)
plt.xlabel("K")
plt.ylabel("Silhouette score")
plt.suptitle("Silhouette Score results on symptoms search popularity in full dimensions")
plt.savefig("SilScoreInFullDims.jpg")
plt.show()
'''

number_of_clusters = 20
dims = 3

kmeans0 = Kmeans(symptom_data)
labels_high_dim = kmeans0.find_clusters(number_of_clusters)

pca_data = PCAPlotting(dataset.d)
reduced_xd = pca_data.reduce_dimensionality(dims)

kmeans = Kmeans(pca_data.reduced_data)
labels_low_dim = kmeans.find_clusters(number_of_clusters)


# pca_data.plot_optimal_pc()
consistency = kmeans.consistency(number_of_clusters, labels_high_dim, labels_low_dim)

print("{ Percentage of points falling in the same cluster together again = ",
      consistency[1], "with dimensions reduced to ", dims, " and number of clusters = ",
                                                           number_of_clusters, " }")

kmeans.plot_kmeans()

temp_dataset.keep_x_symptoms(15)
temp_dataset.merge_regions()
symp_pop = symptomsPopularity.SymptomPopularity(temp_dataset.d)
symp_pop.symptoms_popularity()


