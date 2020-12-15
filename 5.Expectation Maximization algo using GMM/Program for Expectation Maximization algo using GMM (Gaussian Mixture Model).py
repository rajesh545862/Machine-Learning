# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 23:31:44 2020
@author: Rajesh
"""
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Clustering_gmm.csv')

plt.figure(figsize=(7,7))
plt.scatter(data["Weight"],data["Height"])
plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('Data Distribution')
plt.show()

#training k-means model
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)
kmeans.fit(data)
#predictions from kmeans

pred = kmeans.predict(data)
frame = pd.DataFrame(data)
frame['cluster'] = pred
frame.columns = ['Weight', 'Height', 'cluster']

#plotting results

color=['blue','green','cyan', 'black']
for k in range(0,4):data = frame[frame["cluster"]==k]
plt.scatter(data["Weight"],data["Height"],c=color[k])
plt.show()

import pandas as pd
data = pd.read_csv('Clustering_gmm.csv')

# training gaussian mixture model 

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4)
gmm.fit(data)

#predictions from gmm

labels = gmm.predict(data)
frame = pd.DataFrame(data)
frame['cluster'] = labels
frame.columns = ['Weight', 'Height', 'cluster']

color=['blue','green','cyan', 'black']
for k in range(0,4):data = frame[frame["cluster"]==k]
plt.scatter(data["Weight"],data["Height"],c=color[k])
plt.show()


