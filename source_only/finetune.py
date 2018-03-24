"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import os
import os.path as osp
import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator

"""
Configuration Part.
"""

# Path to the textfiles for the trainings and validation set
dataroot = '/scratch0/dataset/domain_generalization/kfold/'
train_file = '../data/sourceonly/art_painting/train.txt'
val_file = '../data/sourceonly/art_painting/val.txt'
out_path = 'results/'

# Learning params
base_learning_rate = 0.0005
num_iters = 45000
batch_size = 64

# Network params
dropout_rate = 0.5
num_classes = 7
scratch_layers = ['fc8']

# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
if not osp.exists(out_path):
    os.makedirs(out_path)

filewriter_path = osp.join(out_path, 'tensorboard')
checkpoint_path = osp.join(out_path, 'checkpoints')

# Create parent path if it doesn't exist
if not osp.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 dataroot=dataroot,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                  dataroot=dataroot,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, scratch_layers)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list1 = [v for v in tf.trainable_variables() if v.name.split('/')[0] not in scratch_layers]
var_list2 = [v for v in tf.trainable_variables() if v.name.split('/')[0] in scratch_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=score,
                                                                  labels=y))
global_step1 = tf.Variable(0, trainable=False)
global_step2 = tf.Variable(0, trainable=False)

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients1 = tf.gradients(loss, var_list1)
    gradients1 = list(zip(gradients1, var_list1))
    gradients2 = tf.gradients(loss, var_list2)
    gradients2 = list(zip(gradients2, var_list2))

    # Create optimizer and apply gradient descent to the trainable variables
    learning_rate1 = tf.train.exponential_decay(base_learning_rate, global_step1,
        15000, 0.96)
    learning_rate2 = tf.train.exponential_decay(base_learning_rate*10, global_step2,
        15000, 0.96)
    optimizer1 = tf.train.GradientDescentOptimizer(learning_rate1)
    optimizer2 = tf.train.GradientDescentOptimizer(learning_rate2)
    
    train_op1 = optimizer1.apply_gradients(grads_and_vars=gradients1, global_step=global_step1)
    train_op2 = optimizer2.apply_gradients(grads_and_vars=gradients2, global_step=global_step2)
    train_op = tf.group(train_op1, train_op2)

# Add gradients to summary
for gradient, var in gradients1:
    tf.summary.histogram(var.name + '/gradient', gradient)
for gradient, var in gradients2:
    tf.summary.histogram(var.name + '/gradient', gradient)
    
# Add the variables we train to the summary
for var in var_list1:
    tf.summary.histogram(var.name, var)
for var in var_list2:
    tf.summary.histogram(var.name, var)


# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))
curr_iter = 0
epoch = 0
best_val_acc = 0

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    while(True):

        if curr_iter>= num_iters:
            break

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)

            # And run the training op
            sess.run(train_op, feed_dict={x: img_batch,
                                          y: label_batch,
                                          keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            curr_iter += 1
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch*train_batches_per_epoch + step)

        
        epoch += 1
        # Validate the model on the entire validation set
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        
        print("{} Iteration number: {}".format(datetime.now(), curr_iter))
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
                                                       
        # save checkpoint of the model
        if best_val_acc<test_acc:
            best_val_acc = test_acc
            checkpoint_name = os.path.join(checkpoint_path, 'model_best.ckpt')
            save_path = saver.save(sess, checkpoint_name)

