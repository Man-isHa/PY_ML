import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano



def train_with_sgd(model, X_train,bv, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            # ADDED! Saving model oarameters
            save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i],bv[i], y_train[i], learning_rate)
            num_examples_seen += 1

#word count threshold = 30	
vocabulary_size = 3010
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "START"
sentence_end_token = "END"


tokens=[]
with open('results_20130124.token', 'r') as f:

	sentences = itertools.chain(*[nltk.sent_tokenize(x.decode('utf-8')) for x in f])
    	# Append SENTENCE_START and SENTENCE_END
    	sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
	
	for l in sentences:	
		token = nltk.word_tokenize(l)
		token = token[:-len(token)+1]+token[4:]	
#		print(token)	
		tokens.append(token)
#		tokens = [nltk.word_tokenize(sent[0].decode('utf-8')) for sent in f]	
		
		
word_freq = nltk.FreqDist(itertools.chain(*tokens))

print "Parsed %d sentences." % (len(sentences))	
print "Found %d unique words tokens." % len(word_freq.items())

vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print "The least frequent word in our vocabulary is '%s' and appeared %d time." % (vocab[-1][0], vocab[-1][1])

for i, sent in enumerate(tokens):
    tokens[i] = [w if w in word_to_index else unknown_token for w in sent]

ctr=0
for i, sent in enumerate(tokens):
	for w in sent:		
		if w == unknown_token:
			ctr +=1
vocab.append(('UNKNOWN_TOKEN',ctr))

h = {}
for k,v in vocab:
	h.update({k:v})

print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentence after Pre-processing: '%s'" % tokens[0]
 

np.save('data/ixtoword.npy', index_to_word)

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokens])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokens])
feat_path = './data/feats.npy'
feats = np.load(feat_path)
dim_embed = 256
dim_hidden = 256
dim_image = 4096
feats = np.load(feat_path)
feats = feats[:158900]

encode_img_W = np.random.uniform(-0.1, 0.1,(dim_image, dim_hidden))
encode_img_b = np.zeros((dim_hidden))

bv = feats*encoding_img_W + encoding_img_b

model = RNNTheano(len(index_to_word), hidden_dim=dim_hidden)
t1 = time.time()
model.sgd_step(X_train[10],bv[10], y_train[10], _LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

if _MODEL_FILE != None:
    load_model_parameters_theano(_MODEL_FILE, model)

train_with_sgd(model, X_train,bv, y_train, nepoch=1000, learning_rate=0.01)
