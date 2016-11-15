# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
# print("Hello Wrld")

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
#from sklearn.linear_model import LogisticRegressionCV
digits= datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))

def sigmoid(x):
    return 1/(1+np.exp(-x))
n_samples=len(digits.images)
data = digits.images.reshape((n_samples, -1))   
X=data[:n_samples]
#print(X)
#X[X<10]=0
#X[X>=10]=1
X=sigmoid(X)

var=X.shape
num_examples = len(X) # training set size
nn_input_dim = var[1]# input layer dimensionality
nn_output_dim = var[1]# output layer dimensionality
## 
### Gradient descent parameters (I picked these by hand)
epsilon = 0.0001 *0.3
beta =0.01
# learning rate for gradient descent
 # regularization strength


p=0.05
    
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
    pj = np.sum(a1,axis=0)/num_examples
    kl = np.sum(p*np.log(p/pj)+(1-p)*np.log((1-p)/(1-pj)))
    data_loss=np.sum(np.sum(np.power((a2-X),2)))/2 + beta*kl
    # Calculating the loss
    # Add regulatization term to loss (optional)
    
    return 1./num_examples * data_loss
    
#def predict(model, x):
#    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
#    # Forward propagation
#    z1 = x.dot(W1) + b1
#    a1 = np.tanh(z1)
#    z2 = a1.dot(W2) + b2
#    exp_scores = np.exp(z2)
#    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
#    return np.argmax(probs, axis=1)


def build_model(nn_hdim, num_passes=10000, print_loss=False):
     
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
   
    # This is what we return at the end
    model = {}
     
    # Gradient descent. For each batch...
    for i in range(0, num_passes):
 
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = sigmoid(z1)
        z2 = a1.dot(W2) + b2
        a2 = sigmoid(z2)
        pj = np.sum(a1,axis=0)/num_examples
        
        mat = np.tile(((-p/pj+(1-p)/(1-pj))),(num_examples,1))
            
        # Backpropagation
        delta3 = a2-X
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = (delta3.dot(W2.T)+beta*mat)* a1*(1-a1)
        dW1= (X.T).dot(delta2)
        db1 = np.sum(delta2, axis=0)
        
       
       
        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
         
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'a2' :a2}
         
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000==0:
          print("Loss after iteration %i: %f",i, calculate_loss(model))
    
    return model
    
    # Build a model with a 3-dimensional hidden layer
model = build_model(75, print_loss=True)
W1=model['W1']
a2=model['a2']
print(W1)
for i in range(W1.shape[1]):
    plt.subplot(2,5,i+1)
    plt.axis('off')
    a=W1[:,i].reshape(8,8)
    plt.imshow(a)
    plt.show
#for i in range(a2.shape[0]):
#    plt.subplot(2,5,i+1)
#    plt.axis('off')
#    a=a2[i,:].reshape(8,8)
#    plt.imshow(a)
#    plt.show
