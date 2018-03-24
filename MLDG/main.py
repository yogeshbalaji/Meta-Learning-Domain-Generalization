import csv
import numpy as np
import pickle
import random
import tensorflow as tf
import os
from data_generator import ImageDataGenerator
from maml import MAML
from tensorflow.python.platform import flags
from tensorflow.contrib.data import Iterator
from random import shuffle

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'PACS', 'PACS')
flags.DEFINE_string('dataroot', '/scratch0/dataset/domain_generalization/kfold/', 'Root folder where PACS dataset is stored')
flags.DEFINE_integer('num_classes', 7, 'number of classes used in classification (e.g. 5-way classification).')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 45000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 64, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.0005, 'the base learning rate of the generator')
flags.DEFINE_float('update_lr', 0.0005, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 3, 'number of inner gradient updates during training.')

## Model options
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', './logs/', 'directory for summaries and checkpoints.') # /scratch0/Projects/new_ideas/domain_generalization/tensorflow_code/mldg/logs
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

def train(model, saver, sess, exp_string, train_file_list, test_file, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 10000
    PRINT_INTERVAL = 100
    TEST_PRINT_INTERVAL = 100
    dropout_rate = 0.5

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = FLAGS.num_classes # for classification, 1 otherwise

    # Defining data loaders
    with tf.device('/cpu:0'):
        tr_data_list = []
        train_iterator_list = []
        train_next_list = []
        
        for i in range(len(train_file_list)):
            tr_data = ImageDataGenerator(train_file_list[i],
                                     dataroot=FLAGS.dataroot,
                                     mode='training',
                                     batch_size=FLAGS.meta_batch_size,
                                     num_classes=num_classes,
                                     shuffle=True)
            tr_data_list.append(tr_data)
            
            train_iterator_list.append(Iterator.from_structure(tr_data_list[i].data.output_types,
                                           tr_data_list[i].data.output_shapes))
            train_next_list.append(train_iterator_list[i].get_next())

        test_data = ImageDataGenerator(test_file,
                                      dataroot=FLAGS.dataroot,
                                      mode='inference',
                                      batch_size=1,
                                      num_classes=num_classes,
                                      shuffle=False)
        
        test_iterator = Iterator.from_structure(test_data.data.output_types,
                                           test_data.data.output_shapes)
        test_next_batch = test_iterator.get_next()


        # create an reinitializable iterator given the dataset structure
        
    # Ops for initializing different iterators
    training_init_op = []
    train_batches_per_epoch = []
    for i in range(len(train_file_list)):
        training_init_op.append(train_iterator_list[i].make_initializer(tr_data_list[i].data))
        train_batches_per_epoch.append(int(np.floor(tr_data_list[i].data_size/FLAGS.meta_batch_size)))
    
    test_init_op = test_iterator.make_initializer(test_data.data)
    test_batches_per_epoch = int(np.floor(test_data.data_size / 1))
    
    
    # Training begins
    
    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
        
        # Sampling training and test tasks
        num_training_tasks = len(train_file_list)
        num_meta_train = num_training_tasks-1
        num_meta_test = num_training_tasks-num_meta_train
        
        # Randomly choosing meta train and meta test domains
        task_list = np.random.permutation(num_training_tasks)
        meta_train_index_list = task_list[:num_meta_train]
        meta_test_index_list = task_list[num_meta_train:]
        
        for i in range(len(train_file_list)):
            if itr%train_batches_per_epoch[i] == 0:
                sess.run(training_init_op[i])
        
        # Populating input tensors

        # Sampling meta train data
        for i in range(num_meta_train):
            
            task_ind = meta_train_index_list[i]
            if i == 0:
                inputa, labela = sess.run(train_next_list[task_ind])
            else:
                inp_tmp, lab_tmp = sess.run(train_next_list[task_ind])
                inputa = np.concatenate((inputa, inp_tmp), axis=0)
                labela = np.concatenate((labela, lab_tmp), axis=0)
        
        inputs_all = list(zip(inputa, labela))
        shuffle(inputs_all)
        inputa, labela = zip(*inputs_all)
        
        # Sampling meta test data
        for i in range(num_meta_test):
            
            task_ind = meta_test_index_list[i]
            if i == 0:
                inputb, labelb = sess.run(train_next_list[task_ind])
            else:
                inp_tmp, lab_tmp = sess.run(train_next_list[task_ind])
                inputb = np.concatenate((inputb, inp_tmp), axis=0)
                labelb = np.concatenate((labelb, lab_tmp), axis=0)
        
        
        feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.KEEP_PROB: dropout_rate}
        
        if itr<FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:    
            input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))


        # Testing periodically
        if itr % TEST_PRINT_INTERVAL == 0:
            test_acc = 0.
            test_loss = 0.
            test_count = 0
            sess.run(test_init_op)
            for it in range(test_batches_per_epoch):	
                
                test_input, test_label = sess.run(test_next_batch)
                
                feed_dict = {model.test_input: test_input, model.test_label: test_label, model.KEEP_PROB: 1.}
                input_tensors = [model.test_loss, model.test_acc]

                result = sess.run(input_tensors, feed_dict)
                test_loss += result[0]
                test_acc += result[1]
                test_count += 1
                
            print('Validation results: Iteration %d, Loss: %f, Accuracy: %f' %(itr, test_loss/test_count, test_acc/test_count))

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))


def main():
    
    # Constructing training and test graphs
    model = MAML()
    model.construct_model_train()
    model.construct_model_test()
    
    model.summ_op = tf.summary.merge_all()
    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
    sess = tf.InteractiveSession()
    
    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)

    resume_itr = 0
    model_file = None

    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    
    print('Loading pretrained weights')
    model.load_initial_weights(sess)

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    filelist_root = '../data/MLDG/'
    domain_dict = {1:'art_painting.txt', 2:'cartoon.txt', 3:'photo.txt', 4:'sketch.txt'}
    train_domain_list = [2, 3, 4]
    test_domain_list = [1]
    
    train_file_list = [os.path.join(filelist_root, domain_dict[i]) for i in train_domain_list]
    test_file_list = [os.path.join(filelist_root, domain_dict[i]) for i in test_domain_list]
    train(model, saver, sess, exp_string, train_file_list, test_file_list[0], resume_itr)

if __name__ == "__main__":
    main()
