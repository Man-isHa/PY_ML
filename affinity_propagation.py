# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 10:58:48 2016

@author: Manisha
"""

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


X = StandardScaler().fit_transform(X)
ap = cluster.AffinityPropagation(damping=.9,preference=-200)

t0 = time.time()
ap.fit(X)
t1 = time.time()
if hasattr(ap, 'labels_'):
    y_pred = ap.labels_.astype(np.int)
else:
    y_pred = ap.predict(X)



        # plot

plt.title('affinity_propagation', size=18)
plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

if hasattr(ap, 'cluster_centers_'):
    centers = ap.cluster_centers_
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
