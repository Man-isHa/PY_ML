import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from sklearn.metrics import confusion_matrix

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def get_data():

	# Reading the training set
	
	with open('./data/nonlinearly separable/class1_train.txt', 'r') as f:
    		lines = f.readlines()
	class1_train = np.zeros((len(lines),4))
	for i in range(len(lines)):
		st = re.findall("[-+]?\d+[\.]?\d*",lines[i])
		for j in range(2):
			class1_train[i][j] = float(st[j])
		class1_train[i][2] = 1
		 
	with open('./data/nonlinearly separable/class2_train.txt', 'r') as f:
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
	
	
		 	
	with open('./data/nonlinearly separable/class1_val.txt', 'r') as f:
    		lines = f.readlines()
	class1_val = np.zeros((len(lines),4))
	for i in range(len(lines)):
		st = re.findall("[-+]?\d+[\.]?\d*",lines[i])
		for j in range(2):
			class1_val[i][j] = float(st[j])
		class1_val[i][2] = 1			 
		
	with open('./data/nonlinearly separable/class2_val.txt', 'r') as f:
    		lines = f.readlines()
	class2_val = np.zeros((len(lines),4))
	for i in range(len(lines)):
		st = re.findall("[-+]?\d+[\.]?\d*",lines[i])
		for j in range(2):
			class2_val[i][j] = float(st[j])
		class2_val[i][3] = 1	

	val = np.concatenate((class1_val,class2_val),axis=0)
	np.random.shuffle(val)
	
		 	
	with open('./data/nonlinearly separable/class1_test.txt', 'r') as f:
    		lines = f.readlines()
	class1_test = np.zeros((len(lines),4))
	for i in range(len(lines)):
		st = re.findall("[-+]?\d+[\.]?\d*",lines[i])
		for j in range(2):
			class1_test[i][j] = float(st[j])
		class1_test[i][2] = 1	
				 
			
	with open('./data/nonlinearly separable/class2_test.txt', 'r') as f:
    		lines = f.readlines()
	class2_test = np.zeros((len(lines),4))
	for i in range(len(lines)):
		st = re.findall("[-+]?\d+[\.]?\d*",lines[i])
		for j in range(2):
			class2_test[i][j] = float(st[j])
		class2_test[i][3] = 1
		

	fig = plt.figure()
	plt.scatter(class1_test[:,0], class1_test[:,1], s=10, c='b', marker="s", label='first')
	plt.scatter(class2_test[:,0], class2_test[:,1], s=10, c='r', marker="o", label='second')
	plt.show()
	
	train = np.concatenate((class1_train,class2_train),axis=0)
	
	test = np.concatenate((class1_test,class2_test),axis=0)
	
	np.random.shuffle(test)
	return train,val,test	 	
    

def build_model():

	
  
    	h1_size = 5
	h2_size = 5
	NLABELS = 2
    	# Symbols
	X = tf.placeholder("float", shape=[None, 2])
    	y = tf.placeholder("float", shape=[None, 2])
	
    	# Weight initializations
	W_1 = tf.Variable(tf.random_normal([2, h1_size]), name='weights')
   	b_1 = tf.Variable(tf.zeros([h1_size], name='bias'))
	W_2 = tf.Variable(tf.random_normal([h1_size, h2_size]), name='weights')
   	b_2 = tf.Variable(tf.zeros([h2_size], name='bias'))
    	W_3 = tf.Variable(tf.random_normal([h2_size, NLABELS]), name='weights')
   	b_3 = tf.Variable(tf.zeros([NLABELS], name='bias'))
	'''
	W_1 = tf.Variable(tf.random_normal([x_size, h1_size]), name='weights')
   	b_1 = tf.Variable(tf.zeros([h1_size], name='bias'))
	W_2 = tf.Variable(tf.random_normal([h1_size, NLABELS]), name='weights')
   	b_2 = tf.Variable(tf.zeros([NLABELS], name='bias'))
    	'''
	# Forward propagation
	h1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)
	h2 = tf.nn.tanh(tf.matmul(h1, W_2) + b_2)
    	yhat = tf.nn.tanh(tf.matmul(h2, W_3) + b_3)
	#yhat = tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(X, W_1) + b_1), W_2) + b_2)
   	predict = tf.argmax(yhat, dimension=1)

    	# Backward propagation
    	cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yhat, y))
    	updates = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
	return X, y, predict, updates, h1, h2, yhat


def main():
	
	checkpoint_file = './hello2.chk'

	train,val,test = get_data()
    	val_X = val[:,0:2]
    	val_y = val[:,2:4]
	
	train_X = train[:,0:2]
	train_y = train[:,2:4]

	test_X = test[:,0:2]
	test_y = test[:,2:4]
	print(test_X.shape, test_y.shape)
    	X, y, predict, updates, h1, h2, yhat = build_model()
	sess = tf.InteractiveSession()
	init = tf.initialize_all_variables()
    	sess.run(init)
    	

    	
    	for epoch in range(101):
        # Train with each example
       	 	for i in range(len(train)):
	 	   	
            		sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        	train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 	sess.run(predict, feed_dict={X: train_X, y: train_y}))
        	val_accuracy  = np.mean(np.argmax(val_y, axis=1) ==
                                 	sess.run(predict, feed_dict={X: val_X, y: val_y}))

        	print("Epoch = %d, train accuracy = %.2f%%, validation accuracy = %.2f%%"
              		% (epoch + 1, 100. * train_accuracy, 100. * val_accuracy))

		if epoch%5 == 0:
			h_1 = sess.run(h1, feed_dict={X: train_X, y: train_y})
			h_2 = sess.run(h2, feed_dict={X: train_X, y: train_y})
			yh = sess.run(yhat, feed_dict={X: train_X, y: train_y})
			#fig = plt.figure()
			#plt.scatter(h_1[:,0], h_1[:,1], s=10, c='b', marker="s")
			#plt.show()
			#plt.scatter(h_2[:,0], h_2[:,1], s=10, c='r', marker="s")
			#plt.show()
			#plt.scatter(yh[:,0], yh[:,1], s=10, c='g', marker="s")
			#plt.show()		
	

	test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 	sess.run(predict, feed_dict={X: test_X, y: test_y}))
	cnf_matrix = confusion_matrix(np.argmax(test_y, axis=1), sess.run(predict, feed_dict={X: test_X, y: test_y}))
	print(cnf_matrix)
	print("test accuracy = %.2f%%" % (100. * test_accuracy))
    	
	saver = tf.train.Saver()
        saver.save(sess, checkpoint_file)
	visualize(val_X, np.nonzero(val_y)[1], val_y,checkpoint_file)
	visualize(test_X, np.nonzero(test_y)[1], test_y,checkpoint_file)
    	

def visualize(a, b ,Y, f_name):
	
	tf.reset_default_graph()
	X, y, predict, updates,h1, h2, yhat = build_model()
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	
        saver.restore(sess, f_name)
	

	x_min, x_max = a[:, 0].min() - .5, a[:, 0].max() + .5
    	y_min, y_max = a[:, 1].min() - .5, a[:, 1].max() + .5
    	h = 0.01
    	# Generate a grid of points with distance h between them
    	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    	# Predict the function value for the whole gid
    	Z = sess.run(predict, feed_dict={X: np.c_[xx.ravel(), yy.ravel()], y: Y})
   	Z = Z.reshape(xx.shape)
    	# Plot the contour and training examples
    	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    	plt.scatter(a[:, 0], a[:, 1], c=b, cmap=plt.cm.Spectral)
	plt.title("2 Hidden layers")
    	plt.show()


if __name__ == '__main__':
	main()
