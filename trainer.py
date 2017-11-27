

# Inport Helper Class - Dataset, to load all images
import dataset

# Inport Tensorflow
import tensorflow as tf

# Import Time and Timedelta, to get actual time lapse measurements
import time
from datetime import timedelta

# Import Math
import math

# Import Random
import random

# Import Numpy
import numpy as np

# Import os
import os

# Disable boring warning messages of which there are so many...
os.environ['TF_CPP_MIN_LOG_LEVEL']='9'

#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# this is the size of my training batches
batch_size = 512

# prepare input data, define classes (This could be automatic)
classes = ['cincuenta_a', 'cincuenta_b', 'diez_a', 'diez_b','dos_pesos_a', 'dos_pesos_b', 'un_peso_a','un_peso_b', 'veinticinco_a', 'veinticinco_b']
num_classes = len(classes)

# 30% of the data will automatically be used for validation
validation_size = 0.3


# Images should be all 128 x 128
img_size = 128

# Images should be RGB, with no Alpha channel (24 Bpp)
num_channels = 3

# Define a path to my training data
train_path='training_data'

# We will load all the training and validation images and labels into memory using openCV and use that during training
# Also at this point do all augmentation in the helper class
# This is where you need a huge memory bank
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)


print("\rComplete reading and augmenting input data.\r")
print("Training set will be :\t{} images.".format(len(data.train.labels)))
print("Validation set will be:\t{} images.".format(len(data.valid.labels)))

# start TF session
session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

# labels go in y_true
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1)

# Network graph params for each layer
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 128

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

# function to create Convolutional Layers
def create_convolutional_layer(input,
               num_input_channels,
               conv_filter_size,
               num_filters):

    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])

    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    ## We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')

    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer



def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer

# function to create fully-connected layer
def create_fc_layer(input,
             num_inputs,
             num_outputs,
             use_relu=True):

    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# fisrt layer
layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)

# second layer
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

# third layer
layer_conv3= create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)

# flattening layer
layer_flat = create_flatten_layer(layer_conv3)

# fully connected layer 1
layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

# fully connected layer 2
layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False)

# softmax prediction (value between 0 and 1 for each class)
y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)

# initialize network
session.run(tf.global_variables_initializer())

# define cross - entropy function -
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)

# use Adam (Adaptive), with custom learning rate
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())

def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "{0} : Training Epoch {1} --- Training Accuracy: {2:>6.1%}, Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}"
    print(msg.format(strftime("%Y-%m-%d %H:%M:%S", gmtime()),epoch + 1, acc, val_acc, val_loss))

total_iterations = 0

# create a saver to use every n iterations
saver = tf.train.Saver()

def train(num_iteration):
    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)


        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        # show progress every now and then
        if i % int(data.train.num_examples/batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, os.getcwd() + os.sep + 'models' + os.sep + 'm1' + os.sep + 'model')


    total_iterations += num_iteration

# train n iterations Epochs = iterations / (images / batch_size)
train(num_iteration=3000)
