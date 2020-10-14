from data import Data
from pca import PCAPlotting
from kmeans import *
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

dataset = Data()
dataset.filter_out_zeros()
dataset.fill_na(0)
symptoms = dataset.d.columns[8:-1]

#these two lines only scale and transform the dataset.d so that it is
#similar to what reduce_dimensionality returns
symptom_data = dataset.d.loc[:,symptoms].values
symptom_data = StandardScaler().fit_transform(symptom_data)
#Commented out since it takes time to produce the graphs
#Finds the optimal number of clusters in full dimensions
#The same thing was done for 2D and 3D yielding the 3 ElbowResults graph

pca_data = PCAPlotting(dataset.d)
reduced_xd = pca_data.reduce_dimensionality(4)

x_vals = []
for i in range(1, 101):
    x_vals.append(i)
res = calculate_WSS(reduced_xd, 100)
plt.plot(x_vals, res)
plt.xlabel("K")
plt.ylabel("WSS")
plt.suptitle("Elbow method results on symptoms search popularity in four dimensions")
plt.savefig("ElbowResIn4Dims.jpg")
plt.show()

'''
number_of_clusters = 20
dims = 4

kmeans0 = Kmeans(symptom_data)
labels_high_dim = kmeans0.find_clusters(number_of_clusters)

pca_data = PCAPlotting(dataset.d)
reduced_xd = pca_data.reduce_dimensionality(dims)

kmeans = Kmeans(pca_data.reduced_data)
labels_low_dim = kmeans.find_clusters(number_of_clusters)


# pca_data.plot_optimal_pc()
consistency = kmeans.consistency(number_of_clusters, labels_high_dim, labels_low_dim)
print(consistency)

print("{ Percentage of points falling in the same cluster together again = ",
      consistency[1], "with dimensions reduced to ", dims, " and number of clusters = ",
                                                           number_of_clusters, " }")

kmeans.plot_kmeans()
'''



