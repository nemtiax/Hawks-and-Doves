import numpy as np
import tensorflow as tf

from ops import *

class MatrixGame(object):
    def __init__(self,model_name,game_matrix,batch_size,z_size=10):
        self.model_name = model_name
        self.game_matrix = game_matrix
        self.matrix_size = game_matrix.shape[0]
        self.batch_size = batch_size
        self.z_size = z_size

    def build(self):
        self.z_input = tf.placeholder(tf.float32,[self.batch_size,self.z_size])
        self.opponent_outputs = tf.placeholder(tf.float32,[self.batch_size,self.matrix_size])
        self.target_distribution = tf.placeholder(tf.float32,[self.matrix_size])

        self.h0 =  tf.nn.sigmoid(linear(self.z_input,10,'h0_linear'))
        self.h1 =  tf.nn.sigmoid(linear(self.h0,10,'h1_linear'))
        self.out = squash(tf.nn.softmax(linear(self.h1,self.matrix_size,'out_linear')))

    def build_loss(self):
        partial_losses = []
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                loss = self.out[i]*self.opponent_outputs[j]*self.game_matrix[i][j]
                partial_losses.append(loss)

        self.adversarial_loss = tf.reduce_sum(partial_losses)

        self.strategy_totals = tf.reduce_sum(self.out,axis=0)/tf.reduce_sum(self.out)
        self.init_loss = js_divergence(self.target_distribution,self.strategy_totals)

    def build_train_ops(self):
        self.adv_train = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.adversarial_loss)
        self.init_train = tf.train.AdamOptimizer(0.002).minimize(self.init_loss)



    def initialize_to_distribution(self,target,batches,sess):
        tf.global_variables_initializer().run()
        for b in range(batches):
            z = np.random.uniform(0,1,size=(self.batch_size,self.z_size))
            _,loss,totals,output = sess.run([self.init_train,self.init_loss,self.strategy_totals,self.out],feed_dict={self.z_input: z,self.target_distribution: target})
            print(loss)
            print(totals)
            #print(output)
        print(output)
        print(totals)

with tf.Session() as sess:
    game_matrix = np.asarray(((-1,1),(0,0.5)))
    mg = MatrixGame("hawks and doves",game_matrix,512,z_size=10)
    mg.build()
    mg.build_loss()
    mg.build_train_ops()
    mg.initialize_to_distribution([0.6,0.4],500,sess)
