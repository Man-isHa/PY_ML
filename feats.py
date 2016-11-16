import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.insert(0,'/home/anant/Documents/caffe/python')
import caffe
caffe.set_mode_gpu()
import pandas as pd
import os
import cPickle
import ipdb
import cv2
import skimage

vgg_model = '/home/anant/Documents/caffe/models/vgg/VGG_ILSVRC_16_layers.caffemodel'
vgg_deploy = '/home/anant/Documents/caffe/models/vgg/VGG_ILSVRC_16_layers_deploy.prototxt'
mean = '/home/anant/Documents/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'

annotation_path = './results_20130124.token'
flickr_image_path = './flickr30k-images/'
feat_path = './data/feats.npy'

annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))

batch_size =10
hid_dim = 4096
features = np.zeros([len(annotations['image'])]+[hid_dim])
iter_until = len(annotations['image']) + batch_size
layers = 'fc7'
width=224
height=224

net = caffe.Net(vgg_deploy, vgg_model, caffe.TEST)
transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(mean).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))




def crop_image(x, target_height=227, target_width=227, as_float=True):
    #image = skimage.img_as_float(skimage.io.imread(x)).astype(np.float32)
    image = skimage.io.imread(x)
    if as_float:
        image = skimage.img_as_float(image).astype(np.float32)

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))


    
     

for start, end in zip(range(0, iter_until, batch_size),range(batch_size, iter_until, batch_size)):

	image_batch_file = annotations['image'][start:end]
        image_batch = np.array(map(lambda x: crop_image(x, target_width=width, target_height=height), image_batch_file))

        caffe_in = np.zeros(np.array(image_batch.shape)[[0,3,1,2]], dtype=np.float32)
	
	net.blobs['data'].reshape(batch_size, 3, height, width)
        for idx, in_ in enumerate(image_batch):
            caffe_in[idx] = transformer.preprocess('data', in_)
	
        out = net.forward_all(blobs=[layers], **{'data':caffe_in})
        feats = out[layers]

        features[start:end] = feats
	
        
np.save(feat_path, features)        
'''
image_batch_file = ['./demo/000542.jpg']
image_batch = np.array(map(lambda x: crop_image(x, target_width=width, target_height=height), image_batch_file))
caffe_in = np.zeros(np.array(image_batch.shape)[[0,3,1,2]], dtype=np.float32)
net.blobs['data'].reshape(batch_size, 3, height, width)
for idx, in_ in enumerate(image_batch):
            caffe_in[idx] = transformer.preprocess('data', in_)
out = net.forward_all(blobs=[layers], **{'data':caffe_in})
feats = out[layers]

np.save('./000542.npy',out[layers])


'''



