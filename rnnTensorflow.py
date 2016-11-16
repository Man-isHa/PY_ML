import re
from mmap import ACCESS_READ, mmap    
import sys
import itertools
import nltk
import numpy as np
from scipy import sparse
from keras.preprocessing import sequence
from datetime import datetime
import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cPickle
import tensorflow.python.platform



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

X_train = X_train[:158900]



model_path = './models'
maxlen = np.max(map(lambda x: len(x), tokens))
dim_embed = 256
dim_hidden = 256
dim_image = 4096
batch_size = 10
n_epochs = 50
n_lstm_steps = maxlen+2 
n_words = len(index_to_word)
annotation_path = './results_20130124.token'
feat_path = './data/feats.npy'

feats = np.load(feat_path)
feats = feats[:158900]





def init_weight(dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

def init_bias(dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)






def build_model():
	
	learning_rate = 0.001
	with tf.device("/cpu:0"):
		Wemb = tf.Variable(tf.random_uniform([n_words, dim_embed], -0.1, 0.1), name='Wemb')
	bemb =init_bias(dim_embed, name='bemb')
	embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')

	encode_img_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_img_W')
	encode_img_b = init_bias(dim_hidden, name='encode_img_b')


	bias_init_vector = np.array([1.0*h[index_to_word[i]] for i in range(len(index_to_word))])
	bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies'
	b=np.zeros((len(bias_init_vector)))
	for i in range(len(bias_init_vector)):
		b[i]=np.log(bias_init_vector[i])	
	bias_init_vector -= np.max(b) 


	if bias_init_vector is not None:
		embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
	else:
	        embed_word_b = init_bias(n_words, name='embed_word_b')


        sentence = tf.placeholder(tf.int32, [batch_size, n_lstm_steps])
	image = tf.placeholder(tf.float32, [batch_size, dim_image])
	mask = tf.placeholder(tf.float32, [batch_size,n_lstm_steps])
	image_emb = tf.matmul(image, encode_img_W) + encode_img_b # (batch_size, dim_hidden)
	lstm = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden)
        state = tf.zeros([batch_size, lstm.state_size])

        loss = 0.0
        with tf.variable_scope("RNN"):
            for i in range(n_lstm_steps): # maxlen + 1
                
		if i>0 : tf.get_variable_scope().reuse_variables()
		if i == 0:
			current_emb = image_emb
		else:
			with tf.device("/cpu:0"):
                		current_emb = tf.nn.embedding_lookup(Wemb, sentence[:,i-1]) + bemb
                
		output, state = lstm(current_emb, state) # (batch_size, dim_hidden)
		labels = tf.expand_dims(sentence[:, i], 1) # (batch_size)
                indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
                concated = tf.concat(1, [indices, labels])
                onehot_labels = tf.sparse_to_dense(
                            concated, tf.pack([batch_size, n_words]), 1.0, 0.0) # (batch_size, n_words)

                logit_words = tf.matmul(output, embed_word_W) + embed_word_b # (batch_size, n_words)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
                cross_entropy = cross_entropy * mask[:,i]#tf.expand_dims(mask, 1)

                current_loss = tf.reduce_sum(cross_entropy)
                loss = loss + current_loss
		train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            loss = loss / tf.reduce_sum(mask[:,1:])
	    
        return loss, sentence, mask, image, train_op


def train():
	
	learning_rate = 0.001
    	momentum = 0.9
	sess = tf.InteractiveSession()
	loss, sentence, mask ,image ,train_op = build_model()
	saver = tf.train.Saver(max_to_keep=50)
    	tf.initialize_all_variables().run()
	for epoch in range(n_epochs):
        	for start, end in zip( \
        	        range(0, len(X_train), batch_size),
        	        range(batch_size, len(X_train), batch_size)
        	        ):
		    current_feats = feats[start:end]
        	    current_captions = X_train[start:end]	
        	    current_caption_matrix = sequence.pad_sequences(current_captions, padding='post', maxlen=maxlen+1)
        	    current_caption_matrix = np.hstack( [np.full( (len(current_caption_matrix),1), 0), current_caption_matrix] ).astype(int)
	
        	    current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
        	    nonzeros = np.array( map(lambda x: (x != 0).sum(), current_caption_matrix ))

        	    for ind, row in enumerate(current_mask_matrix):
        	        row[:nonzeros[ind]] = 1

        	    _, loss_value = sess.run([train_op, loss], feed_dict={
        	        sentence : current_caption_matrix,
			image: current_feats,
        	        mask : current_mask_matrix
        	        })

        	    print "Current Cost: ", loss_value

        	print "Epoch ", epoch, " is done. Saving the model ... "
        	saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
        	learning_rate *= 0.95
train()




def build_generator(maxlen):


	Wemb = tf.Variable(tf.random_uniform([n_words, dim_embed], -0.1, 0.1), name='Wemb')
	bemb =init_bias(dim_embed, name='bemb')
	embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')

	encode_img_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_img_W')
	encode_img_b = init_bias(dim_hidden, name='encode_img_b')


	bias_init_vector = np.array([1.0*h[index_to_word[i]] for i in range(len(index_to_word))])
	bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies'
	b=np.zeros((len(bias_init_vector)))
	for i in range(len(bias_init_vector)):
		b[i]=np.log(bias_init_vector[i])	
	bias_init_vector -= np.max(b) 


	if bias_init_vector is not None:
		embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
	else:
	        embed_word_b = init_bias(n_words, name='embed_word_b')


        sentence = tf.placeholder(tf.int32, [batch_size, n_lstm_steps])
        mask = tf.placeholder(tf.float32, [batch_size,n_lstm_steps])
	image = tf.placeholder(tf.float32, [batch_size, dim_image])
	image_emb = tf.matmul(image, encode_img_W) + encode_img_b # (batch_size, dim_hidden)

        lstm = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden)
	state = tf.zeros([1, lstm.state_size])
        #last_word = image_emb
	image = tf.placeholder(tf.float32, [1, dim_image])
        image_emb = tf.matmul(image, encode_img_W) + encode_img_b

        generated_words = []

        with tf.variable_scope("RNN"):
	    output, state = lstm(image_emb, state)
            last_word = tf.nn.embedding_lookup(Wemb, [1]) + bemb
	    
            for i in range(maxlen):
                tf.get_variable_scope().reuse_variables()

                output, state = lstm(last_word, state)

                logit_words = tf.matmul(output, embed_word_W) + embed_word_b
                max_prob_word = tf.argmax(logit_words, 1)

                with tf.device("/cpu:0"):
                    last_word = tf.nn.embedding_lookup(Wemb, max_prob_word)

                last_word += bemb

                generated_words.append(max_prob_word)

        return generated_words,image






def test(test_feat='./demo/000542.npy', model_path='./models/model-49', maxlen=30): # Naive greedy search

    ixtoword = np.load('data/ixtoword.npy').tolist()
    n_words = len(ixtoword)
    feat = [np.load(test_feat)][0]		
    sess = tf.InteractiveSession()

    generated_words, image = build_generator(maxlen=maxlen)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    generated_word_index= sess.run(generated_words,feed_dict={image:feat})
    generated_word_index = np.hstack(generated_word_index)

    generated_sentence = [ixtoword[x] for x in generated_word_index]
    print(generated_sentence)
test()







