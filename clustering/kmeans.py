import numpy as np
import random
import time
import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn.preprocessing import StandardScaler



np.random.seed(0)
products = range(10000)
users = range(1000)
purchases =[]
for p in range(100000):
    u= random.choice(users)
    p= random.choice(products)
    purchases.append((u,p))
X= purchases;


colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

#Forgy random initialisation chosing k centroids randomly
X = StandardScaler().fit_transform(X)
means = cluster.KMeans(n_clusters=8)
k=8
t0 = time.time()
means.fit(X)
t1 = time.time()
if hasattr(means, 'labels_'):
    y_pred = means.labels_.astype(np.int)
else:
    y_pred = means.predict(X)
print("Cost with random initialisation : ",means.inertia_)

fig1=plt.figure()
plt.title('k_means', size=18)
plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

if hasattr(means, 'cluster_centers_'):
    centers = means.cluster_centers_
    center_colors = colors[:len(centers)]
    plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xticks(())
plt.yticks(())
plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
transform=plt.gca().transAxes, size=15,
horizontalalignment='right')
plt.show()
#kmeans++ maxmin initialisation
means = cluster.KMeans(n_clusters=8,init='k-means++')
t0 = time.time()
means.fit(X)
t1 = time.time()
print("Cost with maximin (k-means++) initialisation : ",means.inertia_)

#Number of time the k-means algorithm will be run with different centroid seeds with n_init
fig2=plt.figure()
plt.title("Mean inertia for various k-means init")
n_init_range = np.array([1, 5, 10, 15, 20])
inertia = np.zeros(5)
for i, n_init in enumerate(n_init_range):
    km = cluster.KMeans(n_clusters=8, n_init=n_init,init='k-means++').fit(X)
    inertia[i] = km.inertia_
p = plt.errorbar(n_init_range, inertia)
plt.xlabel('n_init')
plt.ylabel('inertia')

   
