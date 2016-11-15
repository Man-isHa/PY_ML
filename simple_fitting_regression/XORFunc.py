'''
XOR function using McCulloh- Pitts neuron :

MP neurons are supposed to have a fixed set of weights that are analytically derived hence there was no learning used but I have used weights and biases that have been pre-determined.
Since one layer of weights cannot solve the non-linear problem. I have used another hidden layer so that the whole network functions like an XOR :

x1---(w1,b1)---->|
		 |a----(w3,b3)----->output	
x2---(w2,b2)---->|
'''
import numpy as np


def f(a):
	if a>0:
		return 1
	else:
		return 0	


w1 = [0.5,0.5]
b1 = 0
w2 = [0.5,0.5]
b2 = 0
w3 = [0.25,0.33]
b3 = -0.5

for i in range(4):
	x1 = []
	for i in range(0, 2):
	    x = raw_input('Enter x1 and x2 for (x1 XOR x2)')
	    x1.append(x)
	x1 = map(int,x1)
	x2 = [0,0]
	x2[0] = int(not x1[0])
	x2[1] = int(not x1[1])
	
	a=[0,0]
	a[0] = f(np.dot(np.asarray(x1),np.asarray(w1)) + b1)
	a[1] = f(np.dot(np.asarray(x2),np.asarray(w2)) + b2)
	out = f(np.dot(np.asarray(a),np.asarray(w3)) + b3)

	print(out)


