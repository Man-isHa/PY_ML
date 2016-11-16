import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten,Masking,Reshape
from keras.layers.wrappers import TimeDistributed
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.engine.topology import Merge
from keras.callbacks import ModelCheckpoint

def get_data():
	
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
#			print(token)	
			tokens.append(token)
#			tokens = [nltk.word_tokenize(sent[0].decode('utf-8')) for sent in f]	
		
		
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
	y_train = np.asarray([[word_to_index[w] for w in sent] for sent in tokens])
	index = [i for i in range(len(X_train)) if X_train[i] == []]
	X_train =  [x for x in X_train if x != []]
	y_train =  [x for x in y_train if x != []]
	feat_path = './data/feats.npy'
	feats = np.load(feat_path)
	feats = [feats[i] for i in range(len(feats)) if i not in index]  
	print(len(feats))
	train = [X_train[:153000],feats[:153000],y_train[:153000]]
	val = [X_train[153000:],feats[:153000],y_train[153000:]]
	return train,val	
	




def main():
	train, val = get_data()
	index_to_word = np.load('data/ixtoword.npy').tolist()
	word_index = dict([(w,i) for i,w in enumerate(index_to_word)])
	X_train = np.asarray(train[0])
	feats = np.asarray(train[1])
	y_train = train[2]
	X_train = pad_sequences(X_train, maxlen=83, padding='post')
	y_train = pad_sequences(y_train, maxlen=84, padding='post')
	dim_embed = 256
	dim_hidden = 256
	learning_rate = 0.01
	EMBEDDING_DIM =100
	maxlen = 84	
	#X = np.zeros((len(X_train), maxlen,  len(index_to_word)), dtype=np.bool)
	y = np.zeros((len(y_train), maxlen,  len(index_to_word)), dtype=np.bool)
	#for i, line in enumerate(X_train):
	#	for t, word in enumerate(line):
	#		X[i, t, word] = 1
	for i, line in enumerate(y_train):
		for t, word in enumerate(line):
			y[i, t, word] = 1
			
			
	GLOVE_DIR = './glove.6B/'
	embeddings_index = {}
	f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')																																																																																																																																																																																																																																																																																														
	    embeddings_index[word] = coefs
	f.close()

	print('Found %s word vectors.' % len(embeddings_index))
	ctr= 0
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
		ctr +=1
	        # words not found in embedding index will be all-zeros.
	        embedding_matrix[i] = embedding_vector
	print(ctr)
	print(len(index_to_word))
	image_model = Sequential()
	image_model.add(Dense(256,input_dim=4096))
	image_model.add(Reshape((1,256)))
	print('Build model...')
	lang_model = Sequential()
	lang_model.add(Embedding(len(word_index) + 1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=83,mask_zero=True,trainable=True))
	lang_model.add(Masking(mask_value=0.))
	#lang_model.add(LSTM(256, dropout_W=0.2, return_sequences=True, activation='tanh'))
	lang_model.add(TimeDistributed(Dense(256)))
	model = Sequential()
	model.add(Merge([image_model, lang_model], mode='concat', concat_axis=1)) 
	model.add(LSTM(256, dropout_W=0.2, return_sequences=True, activation='tanh'))
	model.add(TimeDistributed(Dense(len(index_to_word))))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['acc'])
	chkpointer = ModelCheckpoint(filepath='wt_epochend.hdf5',monitor='val_acc',verbose = 1, save_best_only=True, mode='max')
	print('Train...')
#	model.fit([feats, X_train], y, batch_size=10, nb_epoch=100,validation_split=0.1,shuffle='True',callbacks=[chkpointer])
#	model.save_weights('wt_epochend.hdf5')
	model.load_weights('wt_epochend.hdf5')
	dataX=[]
	test_feat='./guitar_player.npy'
	fts = [np.load(test_feat)][0]
	start = np.random.randint(0, len(X_train)-1)
	pattern = X_train[start]
	print "Seed:"
	print "\"", ''.join([index_to_word[value] for value in pattern]), "\""
	# generate characters
	for i in range(1000):
		x = np.reshape(pattern, (1, len(pattern)))
		x = x / float(3010)
		prediction = model.predict([np.asarray(fts),x], verbose=0)
		index = np.argmax(prediction)
		result = index_to_word[index]
		seq_in = [index_to_word[value] for value in pattern]
		sys.stdout.write(result)
		pattern = np.append(pattern, index)
		pattern = pattern[1:len(pattern)]
	print "\nDone."
	

if __name__ == '__main__':			
	main()

#train, val = get_data()






























