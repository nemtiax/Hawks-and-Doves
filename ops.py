import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops



def linear(input_, output_size, scope=None, stddev=0.5, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.random_normal_initializer(stddev=stddev))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def squash(input,factor=0.05):
        return tf.select(input>0.5,(1-factor)+factor*input,input*factor)
        #return 1/(1+tf.exp(-30*(input-0.5)))
