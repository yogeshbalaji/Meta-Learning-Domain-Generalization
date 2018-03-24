""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

## Network helpers
def conv_block(inp, cweight, bweight, stride_y=2, stride_x=2, groups=1):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride = [1, stride_y, stride_x, 1]
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=stride,
                                         padding='SAME')

    if groups==1:
        conv_output = tf.nn.bias_add(convolve(inp, cweight), bweight)
    else:
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=inp)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=cweight)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)
        conv_output = tf.nn.bias_add(conv, bweight)
    
    relu = tf.nn.relu(conv_output)
    
    return relu

def max_pool(x, filter_height, filter_width, stride_y, stride_x,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding)


def lrn(x, radius, alpha, beta, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias)
                                              
def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)
    

def fc(x, wweight, bweight, relu=True):
    """Create a fully connected layer."""
    
    act = tf.nn.xw_plus_b(x, wweight, bweight)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label)
