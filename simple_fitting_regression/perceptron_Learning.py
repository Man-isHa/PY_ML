'''
Rosenblatt's Perceptron learning for linearly separable dataset. I have generated the dataset from make_classification
of sklearn library till I found the linearly separable dataset and saved it. I have the dataset files separately attached  

'''
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import matplotlib
import time


colors = ['red','green']
# x2 - data points , d2 - labels
x = np.loadtxt('x2')	
d = np.loadtxt('d2')
w = np.zeros((2))
b = 0 				# bias
n = 1 				#learning parameter

fig = plt.figure()
plt.ion()
for i in range(len(x)):
	
	y1 = x[i].dot(w) + b
	y = 0.0 if y1<0 else 1.0
	print(y,d[i])
	if y - d[i] < 0:
		w = w + n * x[i]
		b = b + 1
	if y - d[i] > 0:
		w = w - n * x[i]
		b = b - 1
	
	plt.scatter(x[:,0], x[:,1], c=d, cmap=matplotlib.colors.ListedColormap(colors))
	
  a = np.zeros((1,2))
	a[0,0] = -2
	a[0,1] = 2
	b1 = np.zeros((1,2))
	b1[0,0] = -(w[0] * a[0,0] + b)/w[1]
	b1[0,1] = -(w[0] * a[0,1] + b)/w[1]
	plt.plot(a[0,:],b1[0,:])
	plt.draw()
	
	
