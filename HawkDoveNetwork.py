import tensorflow as tf
import numpy as np
from ops import *
class HawkDove(object):
    def __init__(self,model_name,batch_size):
        self.model_name = model_name
        self.v = 1
        self.c = 2
        self.batch_size = batch_size
        self.z1 = tf.placeholder(tf.float32,[self.batch_size,1])
        self.z2 = tf.placeholder(tf.float32,[self.batch_size,1])

    def build(self):
        with tf.variable_scope("player_one"):
            h0_1 = tf.nn.sigmoid(linear(self.z1,10,'h0_1_linear'))
            h1_1 = tf.nn.sigmoid(linear(h0_1,10,'h1_1_linear'))
            self.out1 = tf.nn.sigmoid(linear(h1_1,1,'h2_1_linear'))
        with tf.variable_scope("player_one"):
            h0_2 = tf.nn.sigmoid(linear(self.z2,10,'h0_2_linear'))
            h1_2 = tf.nn.sigmoid(linear(h0_2,10,'h1_2_linear'))
            self.out2 = tf.nn.sigmoid(linear(h1_2,1,'h2_2_linear'))

        return self.out1,self.out2

    def build_loss(self):
        self.loss_1_hh = self.out1*self.out2*(self.v-self.c)/2
        self.loss_1_hd = self.out1*(1-self.out2) * self.v
        self.loss_1_dh = 0
        self.loss_1_dd = (1-self.out1)*(1-self.out2)*(self.v/2)

        self.loss_1 = -tf.reduce_mean(self.loss_1_hh + self.loss_1_hd + self.loss_1_dh + self.loss_1_dd)

        self.loss_2_hh = self.out1*self.out2*(self.v-self.c)/2
        self.loss_2_hd = 0
        self.loss_2_dh = (1-self.out1)*(self.out2) * self.v
        self.loss_2_dd = (1-self.out1)*(1-self.out2)*(self.v/2)

        self.loss_2 = -tf.reduce_mean(self.loss_2_hh + self.loss_2_hd + self.loss_2_dh + self.loss_2_dd)

    def build_train_ops(self):
        t_vars = tf.trainable_variables()

        self.pop1_vars = [var for var in t_vars if 'player_one' in var.name]
        self.pop2_vars = [var for var in t_vars if 'player_two' in var.name]

        self.pop1_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.loss_1, var_list=self.pop1_vars)
        self.pop2_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.loss_2, var_list=self.pop2_vars)


    def train(self,sess):
        tf.global_variables_initializer().run()
        sample_z =  np.random.uniform(0, 1, size=(self.batch_size, 1))


        for epoch in range(50):
            for batch in range(100):
                batch_z1 = np.random.uniform(0, 1, size=(self.batch_size, 1))
                batch_z2 = np.random.uniform(0, 1, size=(self.batch_size, 1))
                sess.run([self.pop1_optim],feed_dict={self.z1: batch_z1,self.z2: batch_z2})

                _,loss_1,loss_2 = sess.run([self.pop2_optim,self.loss_1,self.loss_2],feed_dict={self.z1: batch_z1,self.z2: batch_z2})

                print("Epoch %2d batch %4d: L1 = %.4f, L2 = %.4f" % (epoch,batch,loss_1,loss_2))
            out1 = sess.run(self.out1,feed_dict={self.z1: sample_z})
            out2 = sess.run(self.out2,feed_dict={self.z2: sample_z})
            loss_1 = sess.run(self.loss_1,feed_dict={self.z1: sample_z, self.z2: sample_z})
            loss_1_hh = sess.run(self.loss_1_hh,feed_dict={self.z1: sample_z, self.z2: sample_z})
            print(epoch)
            print(out1)
            print(out2)
            print(loss_1)


with tf.Session() as sess:
    hd = HawkDove("hawks and doves",64)
    hd.build()
    hd.build_loss()
    hd.build_train_ops()
    hd.train(sess)
