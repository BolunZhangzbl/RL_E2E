import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Activation, Input, Dense, GaussianNoise, Lambda, Dropout, Concatenate, Reshape, Add, Embedding, Flatten, Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K



class Actor(tf.keras.Model):

    def __init__(self, msg_total=256, channel=16):
        super(Actor, self).__init__()
        self.m = msg_total
        self.n = channel

        self.actor_inp = Input((1, self.m))
        self.conv0 = Conv1D(filters=self.m, kernel_size=1)
        self.bn0 = BatchNormalization()
        self.ac0 = Activation('elu')

        self.conv1 = Conv1D(filters=self.m, kernel_size=1)
        self.bn1 = BatchNormalization()
        self.ac1 = Activation('elu')

        self.conv2 = Conv1D(filters=2*self.n, kernel_size=1)
        self.bn2 = BatchNormalization()
        self.linear = Activation('linear')

        self.norm = Lambda(lambda x: x / K.sqrt(K.mean(x**2)))

    def call(self, inputs, training=None, mask=None):
        inp = self.actor_inp(inputs)
        x01 = self.conv0(inp)
        x02 = self.bn0(x01)
        x03 = self.ac0(x02)

        x11 = self.conv1(x03)
        x12 = self.bn1(x11)
        x13 = self.ac1(x12)

        x21 = self.con2(x13)
        x22 = self.bn2(x21)
        x23 = self.linear(x22)

        x = self.norm(x23)

        return x







