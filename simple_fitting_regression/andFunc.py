'''
Rosenblatt's Perceptron learning to generate an AND function where the learning parameter(n) is set to 1
and have used ONLINE LEARNING (Weights are updated after each sample passed.)
Trained for 10 epochs to make sure that the set of weights are stable and need no modification

'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

colors = ['red','green']
x = np.array([[0,0],[0,1],[1,0],[1,1]])	
d = np.array([0,0,0,1])
w = np.zeros((2))
b = 0 				# bias
n = 1 				#learning parameter
ep = 1
fig = plt.figure()
plt.ion()
while ep<10:
	for i in range(len(x)):
		y1 = x[i].dot(w) + b
		y = 0 if y1<=0 else 1
		print(y,d[i])
		if y - d[i] < 0:
			w = w + n * x[i]
			b = b + 1
		if y - d[i] > 0:
			w = w - n * x[i]
			b = b - 1
		print(w,b)
		plt.scatter(x[:,0], x[:,1], c=d, cmap=matplotlib.colors.ListedColormap(colors))
	
        	a = np.zeros((1,2))
		a[0,0] = 0
		a[0,1] = 1.5
		b1 = np.zeros((1,2))
		if w[1] == 0 and w[0]!=0:
			b1[0,0] = 1.5
			b1[0,1] = 1.5
			a[0,0] = a[0,1] = -b/w[0]
			plt.plot(a[0,:],b1[0,:])
		if w[1] == 0 and w[0] == 0:
		        print("NO Line :(")
		else:
			b1[0,0] = -(w[0] * a[0,0] + b)/w[1]
			b1[0,1] = -(w[0] * a[0,1] + b)/w[1]
			plt.plot(a[0,:],b1[0,:])
		plt.draw()

	ep = ep+1


