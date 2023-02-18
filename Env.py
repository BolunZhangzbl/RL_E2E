import numpy as np
import gym
from gym import Env, spaces
import tensorflow as tf
from keras.models import Model
from keras.layers import Activation, Input, Dense, GaussianNoise, Lambda, Dropout, Concatenate, Reshape, Add, Embedding, Flatten, Conv1D
from tensorflow.keras.layers import BatchNormalization

from ChannelNet import Channel



##########################################################################
################## Define the Environment Class ##########################
##########################################################################

class E2E_Env(Env):

    def __init__(self, channel_model='AWGN', msg_total=256, channel=16, SNR_train=20):
        self.channel_model = channel_model
        self.channel = channel
        self.msg_total = msg_total
        self.k = np.log2(msg_total)

        # define the observation space and action space
        self.observation_space = spaces.Discrete(msg_total)
        self.action_space = spaces.Box(np.zeros((1, self.channel * 2)), np.ones((1, self.channel * 2)))  # action size: (4,2) = 8

        self.model_cha = Channel('Rayleigh', msg_total=self.msg_total, channel=self.channel)

        self.model_rx = self.create_receiver()
        self.model_rx.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.noise_stddev = np.sqrt(1 / (2 * (10 ** (SNR_train / 10.0))))

    def create_receiver(self):
        rx_inp = Input((1, 4*self.channel))
        conv0 = Conv1D(self.msg_total, kernel_size=1)(rx_inp)
        bn0 = BatchNormalization()(conv0)
        ac0 = Activation('elu')(bn0)

        conv1 = Conv1D(self.msg_total, kernel_size=1)(ac0)
        bn1 = BatchNormalization()(conv1)
        ac1 = Activation('elu')(bn1)

        softmax = Conv1D(self.msg_total, kernel_size=1, activation='softmax')(ac1)

        return Model(rx_inp, softmax)

    # Step function to implement channel and receiver
    def step(self, action, prev_state):
        # raw_input = np.random.randint(0, self.msg_total, (self.batch_size))
        raw_input_next = np.random.randint(0, self.msg_total, (1,))
        label_next = np.zeros((1, self.msg_total))
        label_next[np.arange(1), raw_input_next] = 1
        # self.next_state = raw_input_next
        label_next = np.expand_dims(label_next, axis=1)
        self.next_state = label_next

        signal_rx = self.model_cha(action, self.noise_stddev)
        pred = self.model_rx.predict(signal_rx)

        # Method 1 to calculate loss
        # loss = np.sum(np.square(pred - label), axis=1)
        # self.reward = -loss

        # Method 2 to calculate loss
        # lossscc = tf.keras.losses.SparseCategoricalCrossentropy()
        # lossscc = tf.keras.losses.sparse_categorical_crossentropy
        # loss = lossscc(prev_state, pred)
        # self.reward = -loss

        # Method 3 to calculate loss
        losscc = tf.keras.losses.CategoricalCrossentropy()
        # losscc = tf.keras.losses.categorical_crossentropy
        loss = losscc(prev_state, pred)
        self.reward = -loss

        # Set the threshold to terminate the Episodic Training
        if self.reward >= -0.01:  # 99% accuracy
            self.done = True

        return self.next_state, self.reward, self.done

    def reset(self):
        raw_input = np.random.randint(0, self.msg_total, (1,))
        label = np.zeros((1, self.msg_total))
        label[np.arange(1), raw_input] = 1
        self.state = np.expand_dims(label, axis=1)
        self.reward = 0
        self.done = False

        return self.state

    def cha_noise(self, enc_tuple, sigma):
        enc_tuple = tf.cast(enc_tuple, dtype=tf.float32)
        sig_rx = self.model_cha(enc_tuple, sigma)
        return sig_rx

    def train_rx(self, inputs, outputs):
        history_rx = self.model_rx.fit(inputs, outputs, epochs=1, batch_size=256)
        return history_rx.history['loss'], history_rx.history['accuracy']

    def predict_rx(self, final_signal):
        pred_final_signal = self.model_rx.predict(final_signal)
        return pred_final_signal