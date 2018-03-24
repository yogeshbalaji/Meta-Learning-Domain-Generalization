""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, fc, max_pool, lrn, dropout

FLAGS = flags.FLAGS

class MAML:
    def __init__(self):
        """ must call construct_model() after initializing MAML! """
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.SKIP_LAYER = ['fc8']
        self.forward = self.forward_alexnet
        self.construct_weights = self.construct_alexnet_weights
        self.loss_func = xent
        self.WEIGHTS_PATH = '/scratch0/Projects/new_ideas/domain_generalization/tensorflow_code/pretrained_weights/bvlc_alexnet.npy'
    
    def construct_model_train(self, prefix='metatrain_'):
        # a: training data for inner gradient, b: test data for meta gradient
        
        self.inputa = tf.placeholder(tf.float32)
        self.inputb = tf.placeholder(tf.float32)
        self.labela = tf.placeholder(tf.float32)
        self.labelb = tf.placeholder(tf.float32)
        self.KEEP_PROB = tf.placeholder(tf.float32)
        
        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()

            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            
            num_updates = FLAGS.num_updates
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates

            def task_metalearn(inp, reuse=True):
                """ Function to perform one meta learning update """

                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []
                task_accuraciesb = []

                # Obtaining the gradients on meta train
                task_outputa = self.forward(inputa, weights, reuse=reuse)
                task_lossa = self.loss_func(task_outputa, labela)
                grads = tf.gradients(task_lossa, list(weights.values()))
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))

                # Getting the loss on meta test
                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))
                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                # Populating the metrics
                task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1))
                for j in range(num_updates):
                    task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(labelb, 1)))
                task_output.extend([task_accuracya, task_accuraciesb])

                return task_output

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
            result = task_metalearn((self.inputa, self.inputb, self.labela, self.labelb))
            outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result


        ## Performance & Optimization
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

            self.outputas, self.outputbs = outputas, outputbs
            self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            
            global_step = tf.Variable(0, trainable=False)
            
            var_list1 = [v for v in tf.trainable_variables() if v.name.split('/')[1] not in self.SKIP_LAYER]
            var_list2 = [v for v in tf.trainable_variables() if v.name.split('/')[1] in self.SKIP_LAYER]
            
            learning_rate1 = tf.train.exponential_decay(FLAGS.meta_lr, global_step,
                15000, 0.96)
            learning_rate2 = tf.train.exponential_decay(FLAGS.meta_lr, global_step,
                15000, 0.96)
        
            if FLAGS.pretrain_iterations > 0:

                optimizer1 = tf.train.GradientDescentOptimizer(learning_rate1)
                optimizer2 = tf.train.GradientDescentOptimizer(learning_rate2)

                gradients1 = tf.gradients(self.total_loss1, var_list1) 
                gradients1 = list(zip(gradients1, var_list1))
                gradients2 = tf.gradients(self.total_loss1, var_list2)
                gradients2 = list(zip(gradients2, var_list2))

                train_op1 = optimizer1.apply_gradients(grads_and_vars=gradients1, global_step=global_step)
                train_op2 = optimizer2.apply_gradients(grads_and_vars=gradients2, global_step=global_step)
                self.pretrain_op = tf.group(train_op1, train_op2)
                    
            if FLAGS.metatrain_iterations > 0:
                
                optimizer1 = tf.train.GradientDescentOptimizer(learning_rate1)
                optimizer2 = tf.train.GradientDescentOptimizer(learning_rate2)
                
                gradients1 = tf.gradients(self.total_loss1 + self.total_losses2[FLAGS.num_updates-1], var_list1) 
                gradients1 = list(zip(gradients1, var_list1))
                gradients2 = tf.gradients(self.total_loss1 + self.total_losses2[FLAGS.num_updates-1], var_list2)
                gradients2 = list(zip(gradients2, var_list2))
            
                gradients1 = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in gradients1]
                gradients2 = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in gradients2]
                train_op1 = optimizer1.apply_gradients(grads_and_vars=gradients1, global_step=global_step)
                train_op2 = optimizer2.apply_gradients(grads_and_vars=gradients2, global_step=global_step)
                self.metatrain_op = tf.group(train_op1, train_op2)
        
        ## Summaries
        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)
        tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
            tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])

    
    
    def construct_model_test(self, prefix='test'):
        # a: training data for inner gradient, b: test data for meta gradient
        
        self.test_input = tf.placeholder(tf.float32)
        self.test_label = tf.placeholder(tf.float32)
        
        with tf.variable_scope('model', reuse=None) as testing_scope:
            if 'weights' in dir(self):
                testing_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                raise ValueError('Weights not initilized. Create training model before testing model')

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            losses, outputs = [], []
            accuracies = []
            num_updates = 1

            outputs = self.forward(self.test_input, weights) # reuse is used for normalization. Need not be used now since there is no BN in Alexnet
            losses = self.loss_func(outputs, self.test_label)
            accuracies = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputs), 1), tf.argmax(self.test_label, 1))
            
        self.test_loss = losses
        self.test_acc = accuracies
        
    
    def construct_alexnet_weights(self):
        weights = {}
        
        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3

        with tf.variable_scope('conv1') as scope:
            weights['conv1_weights'] = tf.get_variable('weights', shape=[11, 11, 3, 96])
            weights['conv1_biases'] = tf.get_variable('biases', [96])
        
        with tf.variable_scope('conv2') as scope:
            weights['conv2_weights'] = tf.get_variable('weights', shape=[5, 5, 48, 256])
            weights['conv2_biases'] = tf.get_variable('biases', [256])
        
        with tf.variable_scope('conv3') as scope:
            weights['conv3_weights'] = tf.get_variable('weights', shape=[3, 3, 256, 384])
            weights['conv3_biases'] = tf.get_variable('biases', [384])
        
        with tf.variable_scope('conv4') as scope:
            weights['conv4_weights'] = tf.get_variable('weights', shape=[3, 3, 192, 384])
            weights['conv4_biases'] = tf.get_variable('biases', [384])
        
        with tf.variable_scope('conv5') as scope:
            weights['conv5_weights'] = tf.get_variable('weights', shape=[3, 3, 192, 256])
            weights['conv5_biases'] = tf.get_variable('biases', [256])
        
        with tf.variable_scope('fc6') as scope:
            weights['fc6_weights'] = tf.get_variable('weights', shape=[9216, 4096])
            weights['fc6_biases'] = tf.get_variable('biases', [4096])
            
        with tf.variable_scope('fc7') as scope:
            weights['fc7_weights'] = tf.get_variable('weights', shape=[4096, 4096])
            weights['fc7_biases'] = tf.get_variable('biases', [4096])
        
        with tf.variable_scope('fc8') as scope:
            weights['fc8_weights'] = tf.get_variable('weights', shape=[4096, 7], initializer=fc_initializer)
            weights['fc8_biases'] = tf.get_variable('biases', [7])
        
        return weights

    def load_initial_weights(self, session):
        """Load weights from file into network.

        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope('model', reuse=True):
		            with tf.variable_scope(op_name, reuse=True):

		                for data in weights_dict[op_name]:

		                    # Biases
		                    if len(data.shape) == 1:
		                        var = tf.get_variable('biases', trainable=True)
		                        session.run(var.assign(data))

		                    # Weights
		                    else:
		                        var = tf.get_variable('weights', trainable=True)
		                        session.run(var.assign(data))

    
    
    def forward_alexnet(self, inp, weights, reuse=False, scope=''):
        # reuse is for the normalization parameters.
        
        conv1 = conv_block(inp, weights['conv1_weights'], weights['conv1_biases'], stride_y=4, stride_x=4, groups=1)
        #norm1 = lrn(conv1, 2, 1e-05, 0.75)
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID')
        
        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv_block(pool1, weights['conv2_weights'], weights['conv2_biases'], stride_y=1, stride_x=1, groups=2)
        #norm2 = lrn(conv2, 2, 1e-05, 0.75)
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID')
        
        # 3rd Layer: Conv (w ReLu)
        conv3 = conv_block(pool2, weights['conv3_weights'], weights['conv3_biases'], stride_y=1, stride_x=1, groups=1)

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv_block(conv3, weights['conv4_weights'], weights['conv4_biases'], stride_y=1, stride_x=1, groups=2)

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv_block(conv4, weights['conv5_weights'], weights['conv5_biases'], stride_y=1, stride_x=1, groups=2)
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fc(flattened, weights['fc6_weights'], weights['fc6_biases'], relu=True)
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, weights['fc7_weights'], weights['fc7_biases'], relu=True)
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations
        fc8 = fc(dropout7, weights['fc8_weights'], weights['fc8_biases'], relu=False)
        
        return fc8


