import os
import os.path as osp
import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator

# Path to the textfiles for the trainings and validation set
dataroot = '/scratch0/dataset/domain_generalization/kfold/'
test_file = '../data/sourceonly/art_painting/test.txt'
checkpoint_path = 'results/checkpoints'
num_classes = 7
scratch_layers = ['fc8']
batch_size = 1

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    test_data = ImageDataGenerator(test_file,
                                  dataroot=dataroot,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(test_data.data.output_types,
                                       test_data.data.output_shapes)
    next_batch_test = iterator.get_next()

# Ops for initializing the two different iterators
test_init_op = iterator.make_initializer(test_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, scratch_layers)

# Link variable to model output
score = model.fc8

# Get the number of training/validation steps per epoch
test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    saver.restore(sess, os.path.join(checkpoint_path, 'model_best.ckpt'))

    # Add the model graph to TensorBoard
    # Testing the best val model

    print("{} Start testing on new domain".format(datetime.now()))
    sess.run(test_init_op)
    test_acc = 0.
    test_count = 0
    for _ in range(test_batches_per_epoch):

        img_batch, label_batch = sess.run(next_batch_test)
        acc = sess.run(accuracy, feed_dict={x: img_batch,
                                            y: label_batch,
                                            keep_prob: 1.})
        test_acc += acc
        test_count += 1
    test_acc /= test_count
    print("{} Test Accuracy = {:.4f}".format(datetime.now(),
                                                   test_acc))
