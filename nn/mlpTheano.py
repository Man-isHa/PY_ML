import theano.tensor as T
import numpy as np
import os
import matplotlib.pyplot as plt
import re



def get_data():

	# Reading the training set
	
	with open('./data/linearly separable/class1_train.txt', 'r') as f:
    		lines = f.readlines()
	class1_train = np.zeros((len(lines),4))
	for i in range(len(lines)):
		st = re.findall("[-+]?\d+[\.]?\d*",lines[i])
		for j in range(2):
			class1_train[i][j] = float(st[j])
		class1_train[i][2] = 1
		 
	with open('./data/linearly separable/class2_train.txt', 'r') as f:
    		lines = f.readlines()
	class2_train = np.zeros((len(lines),4))
	for i in range(len(lines)):
		st = re.findall("[-+]?\d+[\.]?\d*",lines[i])
		for j in range(2):
			class2_train[i][j] = float(st[j])
		class2_train[i][3] = 1		
		

	fig = plt.figure()
	plt.scatter(class1_train[:,0], class1_train[:,1], s=10, c='b', marker="s", label='first')
	plt.scatter(class2_train[:,0], class2_train[:,1], s=10, c='r', marker="o", label='second')
	plt.show()
	
	train = np.concatenate((class1_train,class2_train),axis=0)
	np.random.shuffle(train)
	
	
		 	
	with open('./data/linearly separable/class1_val.txt', 'r') as f:
    		lines = f.readlines()
	class1_val = np.zeros((len(lines),4))
	for i in range(len(lines)):
		st = re.findall("[-+]?\d+[\.]?\d*",lines[i])
		for j in range(2):
			class1_val[i][j] = float(st[j])
		class1_val[i][2] = 1			 
		
	with open('./data/linearly separable/class2_val.txt', 'r') as f:
    		lines = f.readlines()
	class2_val = np.zeros((len(lines),4))
	for i in range(len(lines)):
		st = re.findall("[-+]?\d+[\.]?\d*",lines[i])
		for j in range(2):
			class2_val[i][j] = float(st[j])
		class2_val[i][3] = 1	

	val = np.concatenate((class1_val,class2_val),axis=0)
	np.random.shuffle(val)
	
		 	
	with open('./data/linearly separable/class1_test.txt', 'r') as f:
    		lines = f.readlines()
	class1_test = np.zeros((len(lines),4))
	for i in range(len(lines)):
		st = re.findall("[-+]?\d+[\.]?\d*",lines[i])
		for j in range(2):
			class1_test[i][j] = float(st[j])
		class1_test[i][2] = 1				 
			
	with open('./data/linearly separable/class2_test.txt', 'r') as f:
    		lines = f.readlines()
	class2_test = np.zeros((len(lines),4))
	for i in range(len(lines)):
		st = re.findall("[-+]?\d+[\.]?\d*",lines[i])
		for j in range(2):
			class2_test[i][j] = float(st[j])
		class2_test[i][3] = 1	
	
	test = np.concatenate((class1_test,class2_test),axis=0)
	np.random.shuffle(test)
	
	return train,val,test	 	

def __init__(self, input, n_in, n_out):
	# initialize with 0 the weights W as a matrix of shape (n_in, n_out)
	self.W = theano.shared(value=numpy.zeros((n_in, n_out),dtype=theano.config.floatX),name=’W’,borrow=True)
	# initialize the biases b as a vector of n_out 0s
	self.b = theano.shared(value=numpy.zeros((n_out,),dtype=theano.config.floatX),name=’b’,borrow=True)
	self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
	self.y_pred = T.argmax(self.p_y_given_x, axis=1)
	self.params = [self.W, self.b]
	self.input = input

def negative_log_likelihood(self, y):
	return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    

