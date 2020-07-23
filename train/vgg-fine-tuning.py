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
# from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling2D
# from keras.optimizers import SGD
import cv2, numpy as np
import tensorflow as tf


# In[6]:


img_width, img_height = 224, 224

top_model_weights_path = '128_128_2_2_4classes.h5'#bottleneck_fc_model_multi_40epoch_6class_2lyr.h5'
train_data_dir = 'pix3d-ML2/train'
validation_data_dir = 'pix3d-ML2/validation'
test_data_dir = 'pix3d-ML2/test'
botfet_train = 'bottleneck_features_train.npy'
botfet_validation = 'bottleneck_features_validation.npy'
num_classes = 4
# t = [1049,914,1097,1245,1056,1743]
# z = [249,191,254,244,244,354] #sum must be div by 16
t = [920,1097,1056,1743]#1049,1245,
z = [192,254,244,350] #sum must be div by 16 #249,244,
nb_train_samples = sum(t)
nb_validation_samples = sum(z)
epochs = 30
batch_size = 64


# In[3]:


# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255)
#     shear_range=0.3,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


# In[15]:


# y_train = np.load(open('label_train.npy','rb'))
y_train = np.array([0]*z[0] + [1]*z[1] + [2]*z[2] + [3]*z[3])
# print(y_train)
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)

cl = {}
for i in range(len(class_weights)):
    cl[i] = class_weights[i]
print(cl)
# cl={0:1.12869399,1:1.29540481,2:1.0793072,3:0.95100402,4:1.12121212,5:0.67928858}


# In[7]:


# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# server = tf.train.Server.create_local_server()
# sess = tf.Session(server.target)

# from keras import backend as K
# K.set_session(sess)
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    # build the VGG16 network
    model = applications.VGG16(weights='imagenet', include_top=False,input_shape = (224,224,3))
    print('Model saved.')
    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    print(model.output_shape[1:],model.output_shape)
#     top_model.add(Flatten(input_shape=model.output_shape[1:]))
#     top_model.add(Dense(256, activation='relu'))
#     top_model.add(Dropout(0.5))
#     top_model.add(Dense(num_classes, activation='softmax'))
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dropout(0.2))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dropout(0.3))
    top_model.add(Dense(num_classes, activation='softmax'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
    mb = Sequential()
    mb.add(model)
    mb.add(top_model)
    # model.add(top_model)
    model = mb
#     for layer in model.layers[:25]:
#         layer.trainable = False
    
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-5, momentum=0.9),
              metrics=['accuracy'])


model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    class_weight=cl)

model.save_weights('fin_ep30_bt64_bed_book_ch_desk_sof_tb_4class.h5')


# In[8]:


def only_load():
    model = applications.VGG16(weights='imagenet', include_top=False,input_shape = (224,224,3))
    print('Model saved.')
    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    print(model.output_shape[1:],model.output_shape)
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(num_classes, activation='softmax'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    mb = Sequential()
    mb.add(model)
    mb.add(top_model)
    # model.add(top_model)
#     model = mb
    mb.load_weights('fin_ep50_bt64_bed_book_ch_desk_sof_tb.h5')
    return mb


# In[8]:


labels = (train_generator.class_indices)
print(labels)


# In[6]:


import os
from PIL import Image,ImageFilter

model = only_load()

pos = [0,0,0,0,0,0] #chairmatch,chairmismatch,tablematch,tablemismatch
neg = [0,0,0,0,0,0]
for r,d,f in os.walk(test_data_dir):
    for files in f:
        rr = r.split('/')
#         print(r+'/'+files)
        if 'bed' in rr[2]:
            image2 = np.asarray(Image.open(r+'/'+files))
            image2 = image2[:,:,0:3]
#             print(image2.shape)
            image2 = np.reshape(image2,[1,224,224,3])
            classes = model.predict_classes(image2)

            if classes[0]==0:
                pos[0]+=1
            else:
                neg[0]+=1
                
        elif 'bookcase' in rr[2]:

            image2 = np.asarray(Image.open(r+'/'+files))
            image2 = image2[:,:,0:3]
#             print(image2.shape)
            image2 = np.reshape(image2,[1,224,224,3])
            classes = model.predict_classes(image2)
#             print(classes)
            if classes[0]==1:
                pos[1]+=1
            else:
                neg[1]+=1
#             break
        elif 'chair' in rr[2]:
#             print(r+'/'+files)
            image2 = np.asarray(Image.open(r+'/'+files))
            image2 = image2[:,:,0:3]
#             print(image2.shape)
            image2 = np.reshape(image2,[1,224,224,3])
            classes = model.predict_classes(image2)
#             print(classes)
            if classes[0]==2:
                pos[2]+=1
            else:
                neg[2]+=1
        elif 'desk' in rr[2]:
#             print(r+'/'+files)
            image2 = np.asarray(Image.open(r+'/'+files))
            image2 = image2[:,:,0:3]
#             print(image2.shape)
            image2 = np.reshape(image2,[1,224,224,3])
            classes = np.argmax(model.predict(image2))
            ll = model.predict(image2).sort()
#             print(classes)
            if classes[0]==3:
                pos[3]+=1
            else:
                neg[3]+=1
        elif 'sofa' in rr[2]:
#             print(r+'/'+files)
            image2 = np.asarray(Image.open(r+'/'+files))
            image2 = image2[:,:,0:3]
#             print(image2.shape)
            image2 = np.reshape(image2,[1,224,224,3])
            classes = model.predict_classes(image2)
#             print(classes)
            if classes[0]==4:
                pos[4]+=1
            else:
                neg[4]+=1
        else:
            image2 = np.asarray(Image.open(r+'/'+files))
            image2 = image2[:,:,0:3]
#             print(image2.shape)
            image2 = np.reshape(image2,[1,224,224,3])
            classes = model.predict_classes(image2)
#             print(classes)
            if classes[0]==5:
                pos[5]+=1
            else:
                neg[5]+=1
print('done')


# In[9]:


import os
from PIL import Image,ImageFilter

# model = only_load()

pos = [0,0,0,0] #chairmatch, chairmismatch, tablematch, tablemismatch
neg = [0,0,0,0]
comp = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
#what is the highest wrong class for each class
mapp = {'bookcase': 0, 'chair': 1, 'sofa': 2, 'table': 3}
for r,d,f in os.walk(test_data_dir):
    for files in f:
        rr = r.split('/')
        idr = mapp[rr[2]]
        
        image2 = np.asarray(Image.open(r+'/'+files))
        image2 = image2[:,:,0:3]
#             print(image2.shape)
        image2 = np.reshape(image2,[1,224,224,3])
        classes = model.predict(image2)[0]
#             classes = classes[0]
#             print(classes)
        if sorted(classes)[-2]!=0:
            sr = [sorted(classes)[-2],sorted(classes)[-1]]
        else:
            sr = [sorted(classes)[-1]]
#         print('cls:',classes,sr,idr,rr)
        
        if classes[idr] in sr:
            pos[idr]+=1
            if classes[idr]!=sr[-1]:
#                 print(classes,np.where(classes==sr[-1])[0][0])
                comp[idr][np.where(classes==sr[-1])[0][0]]+=1
        else:
            neg[idr]+=1
                
        
print('done')


# In[12]:


print(pos,neg,comp)


# In[14]:


for en,item in enumerate(pos):
    print(item/(item+neg[en]))
    #128,128,2,2 all layer unfrozen


# In[11]:


for en,item in enumerate(pos):
    print(item/(item+neg[en]))
    


# In[21]:


for en,item in enumerate(pos):
    print(item/(item+neg[en]))
    # 128 128 2 2 layer frozen


# In[50]:


for en,item in enumerate(pos):
    print(item/(item+neg[en]))


# In[8]:


for en,item in enumerate(pos):
    print(item/(item+neg[en]))
    #128,128,2,2 all layer unfreeze e-5,40epoch


# In[ ]:




