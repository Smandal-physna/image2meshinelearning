#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, Conv2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import multi_gpu_model
from sklearn.utils import class_weight
from matplotlib import pyplot as plt
import cv2, numpy as np
import tensorflow as tf
import sys
import os
import numpy


img_width, img_height = 224, 224

top_model_weights_path = 'fin_ep30_bt64_bed_book_ch_desk_sof_tb_4class.h5'
num_classes = 4
subclass_check = 'subclass-check'

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
	model = applications.VGG16(weights='imagenet', include_top=False,input_shape = (img_height,img_width,3))
	print('Model saved.')
# build a classifier model to put on top of the convolutional model
	top_model = Sequential()
	print(model.output_shape[1:],model.output_shape)
	top_model.add(Flatten(input_shape=model.output_shape[1:]))
	top_model.add(Dense(128, activation='relu'))
	top_model.add(Dropout(0.2))
	top_model.add(Dense(128, activation='relu'))
	top_model.add(Dropout(0.2))
	top_model.add(Dense(num_classes, activation='softmax'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
#	top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
	mb = Sequential()
	mb.add(model)
	mb.add(top_model)
# model.add(top_model)
#     model = mb
	mb.load_weights(top_model_weights_path)

# In[8]:


lls = sorted(os.listdir(subclass_check))
labels = {}
for item in range(len(lls)):
	labels[item]=lls[item]
print(labels)

image2 = cv2.imread(sys.argv[1])
if image2.shape[0]!=img_height or image2.shape[1]!=img_width:
    image2 = cv2.resize(image2,(img_height,img_width))
print(image2.shape)
img2 = image2
image2 = np.reshape(image2,[1,img_height,img_width,3])
#exit(0)
classes = mb.predict(image2)[0]
print(classes)
cl = np.argmax(classes)
#cl = sorted(classes)[-1]
#print(classes.shape,model.predict(image2).shape)
print('************************** The furniture is: ',labels[cl],' !! ***************************')
#print(classes)
#for it in range(len(classes)):
#	print(labels[it],':',sortedclasses[it])


############ FIND SUB-CLASS ON FOUND SUPER CLASS
cand = []
inlist = []
for r,d,f in os.walk(subclass_check+'/'+labels[cl]):
    for files in f:
#        print(files,r)
        if not ('png' in files):
            continue
        img1 = numpy.asarray(cv2.imread(r+'/'+files))
        #surf=cv2.xfeatures2d.SURF_create()
        img2 = cv2.resize(img2,(128,128))
        # find the keypoints and descriptors with SURF
        #kp1, des1 = surf.detectAndCompute(img1,None)
        #kp2, des2 = surf.detectAndCompute(img2,None)
        orb = cv2.ORB_create()
        #print(img1.shape,img2.shape)
        kp1, desc1 = orb.detectAndCompute(img1, None)
        kp2, desc2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        similar_regions = [i for i in matches if i.distance < 70]
        if len(matches) == 0:
            continue
        else:
            percent = len(similar_regions) / len(matches)
            #print(percent)
            if r.split('/')[2] not in inlist:
                inlist.append(r.split('/')[2])
                cand.append([percent,r+'/'+files])

sorted(cand, key=lambda x: x[0])
print("subfile:",cand)
print(cand[-1])

