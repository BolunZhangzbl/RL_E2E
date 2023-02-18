import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Activation, Input, Dense, GaussianNoise, Lambda, Dropout, Concatenate, Reshape, Add, Embedding, Flatten, Conv1D
from tensorflow.keras.layers import BatchNormalization



class Critic(tf.keras.Model):
    def __init__(self, msg_total, channel):
        super(Critic, self).__init__()
        self.m = msg_total
        self.n = channel

        self.state_inp = Input((1, self.m))
        self.action_inp = Input((1, self.n))

        self.d0 = Dense(32, activation='elu')
        self.d1 = Dense(32, activation='elu')
        self.d2 = Dense(self.m, activation='elu')
        self.d3 = Dense(self.m, activation='elu')
        self.linear = Dense(self.m, activation='linear')

        self.flat0 = Flatten()
        self.flat1 = Flatten()

        self.bn0 = BatchNormalization()
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.concat = Concatenate()

    def call(self, s_inp, a_inp, training=None, mask=None):
        c00 = self.state_inp(s_inp)
        c01 = self.flat0(c00)
        c02 = self.d0(c01)

        c10 = self.action_inp(a_inp)
        c11 = self.flat1(c10)
        c12 = self.d1(c11)

        c2 = self.concat([c02, c12])

        c31 = self.d2(c2)
        c32 = self.bn0(c31)
        c33 = self.d3(c32)
        c34 = self.bn1(c33)
        c35 = self.linear(c34)
        c36 = self.bn2(c35)

        return c36



