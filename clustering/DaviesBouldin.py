from __future__ import print_function
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt

import numpy as np
import random

from sklearn.metrics.pairwise import euclidean_distances


products = range(10000)
users = range(1000)
purchases =[]
for i in range(100000):
    u= random.choice(users)
    p= random.choice(products)
    purchases.append((u,p))

X= np.asarray(purchases);
# normalize dataset for easier parameter selection
X = StandardScaler().fit_transform(X)
#sa = np.zeros(11) : for silhouette
val=np.arange(1,12,1)
res=np.zeros(11)
k=1
n_clusters=1
p=-1
while (k<=11):
    n_clusters +=1

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters

    #silhouette_avg = silhouette_score(X, cluster_labels)
    #print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    
    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    

    # Davies Bouldin
    y_pred = cluster_labels
    k = n_clusters
    y_pred = np.asarray(y_pred)
    average = np.zeros(k)
    maxx = np.zeros(k)
    for i in range(0,k):
        summ=0
        index=np.r_[(y_pred==i).nonzero()].tolist()
        for j in index:
            summ+=euclidean_distances(X[j,:],centers[i,:])
        average[i]=summ/len(index)
    for i in range(0,k):
        for j in range(i+1,k):
            db=(average[i]+average[j])/euclidean_distances(centers[i,:],centers[j,:])
            if maxx[i]<db:
                maxx[i]=db
    p=p+1
    res[p]=np.mean(maxx)
    #sa[p]=silhouette_avg
#print("Optimal Silhouette  ",np.argmax(sa)+2)
print("Optimal Davies Boulden  ",np.argmin(res)+2)
plt.plot(val,res)
plt.show()
