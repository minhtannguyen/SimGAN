"""
Implementation of `3.1 Appearance-based Gaze Estimation` from
[Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/pdf/1612.07828v1.pdf).

Note: Only Python 3 support currently.
"""

import os
import sys
import keras
from keras import applications
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
import h5py
import numpy as np
import tensorflow as tf

print('tf-version',tf.__version__, 'keras-version', keras.__version__)

data_dir = '/mnt/data/eye-gaze'
cache_dir = '.'

# load the data file and extract dimensions
with h5py.File(os.path.join(data_dir,'gaze.h5'),'r') as t_file:
    print(list(t_file.keys()))
    assert 'image' in t_file, "Images are missing"
    assert 'look_vec' in t_file, "Look vector is missing"
    assert 'path' in t_file, "Paths are missing"
    print('Synthetic images found:',len(t_file['image']))
    for _, (ikey, ival) in zip(range(1), t_file['image'].items()):
        print('image',ikey,'shape:',ival.shape)
        img_height, img_width = ival.shape
        img_channels = 1
    syn_image_stack = np.stack([np.expand_dims(a,-1) for a in t_file['image'].values()],0)

with h5py.File(os.path.join(data_dir,'real_gaze.h5'),'r') as t_file:
    print(list(t_file.keys()))
    assert 'image' in t_file, "Images are missing"
    print('Real Images found:',len(t_file['image']))
    for _, (ikey, ival) in zip(range(1), t_file['image'].items()):
        print('image',ikey,'shape:',ival.shape)
        img_height, img_width = ival.shape
        img_channels = 1
    real_image_stack = np.stack([np.expand_dims(a,-1) for a in t_file['image'].values()],0)

#
# training params
#

nb_steps = 20 # originally 10000, but this makes the kernel time out
batch_size = 49
k_d = 1  # number of discriminator updates per step
k_g = 2  # number of generative network updates per step
log_interval = 100
pre_steps = 15 # for pretraining

import ipdb; ipdb.set_trace()