import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Activation, Input, Dense, GaussianNoise, Lambda, Dropout, Concatenate, Reshape, Add, Embedding, Flatten, Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

class Channel():
    def __init__(self, mode='Rayleigh', msg_total=256, channel=8):
        super(Channel, self).__init__()
        assert mode in ('AWGN', 'Rayleigh', 'Rician')
        self.mode = mode
        self.m = msg_total
        self.n = channel

        self.rician_factor = 1

    #@tf.function
    def __call__(self, x, noise_sigma):
        if self.mode == 'AWGN':
            noise = K.random_normal(K.shape(x), mean=0, stddev=noise_sigma*np.sqrt(1/2))
            y = Add()([x, noise])
            return y

        if self.mode == 'Rayleigh':
            x = K.concatenate([x[:, :, :self.n], x[:, :, self.n:]], axis=1)

            # Calculating the impaired signal
            H_R = K.random_normal(K.shape(x[:, 0, :]), 0, np.sqrt(1 / 2))  # np.sqrt(1/2)
            H_I = K.random_normal(K.shape(x[:, 1, :]), 0, np.sqrt(1 / 2))
            real = H_R * x[:, 0, :] - H_I * x[:, 1, :]
            imag = H_R * x[:, 1, :] + H_I * x[:, 0, :]
            noise_r = K.random_normal(K.shape(real), mean=0, stddev=noise_sigma)
            noise_i = K.random_normal(K.shape(imag), mean=0, stddev=noise_sigma)
            real = Add()([real, noise_r])
            imag = Add()([imag, noise_i])

            # Calculating the estimated channel response
            y = K.concatenate([real, imag], axis=-1)

            h_real = x[:, 0, :] * real + x[:, 1, :] * imag
            h_imag = x[:, 0, :] * imag - x[:, 1, :] * real
            h_real /= (x[:, 0, :] ** 2 + x[:, 1, :] ** 2)
            h_imag /= (x[:, 0, :] ** 2 + x[:, 1, :] ** 2)
            h_hat = K.concatenate([h_real, h_imag], axis=-1)

            # Concatenate the impaired signal and the channel response
            results = K.concatenate([y, h_hat], axis=-1)
            results = tf.expand_dims(results, axis=1)

            return results

        if self.mode == 'Rician':
            x = K.concatenate([x[:, :, :self.n], x[:, :, self.n:]], axis=1)

            k = 10 ** (self.rician_factor / 10.0)  # ration of LOS to scattered components
            mu = np.sqrt(k / (k + 1))  # mean
            s = np.sqrt(1 / 2 * (k + 1))  # variance
            C_R = s * K.random_normal(K.shape(x[:, 0, :]), 0, np.sqrt(1 / 2)) + mu
            C_I = s * K.random_normal(K.shape(x[:, 1, :]), 0, np.sqrt(1 / 2))

            noise_r = K.random_normal(K.shape(x[:, 0, :]), 0, noise_sigma)
            noise_i = K.random_normal(K.shape(x[:, 1, :]), 0, noise_sigma)
            real = C_R * x[:, 0, :] - C_I * x[:, 1, :] + noise_r
            imag = C_R * x[:, 1, :] + C_I * x[:, 0, :] + noise_i

            # Concatenate the real and imaginary parts of received signal
            y = K.concatenate([real, imag], axis=-1)

            # Calculating the estimated channel response
            c_real = x[:, 0, :] * real + x[:, 1, :] * imag
            c_imag = x[:, 0, :] * imag - x[:, 1, :] * real
            c_real /= (x[:, 0, :] ** 2 + x[:, 1, :] ** 2)
            c_imag /= (x[:, 0, :] ** 2 + x[:, 1, :] ** 2)
            c_hat = K.concatenate([c_real, c_imag], axis=-1)

            # Concatenate the impaired signal and the estimated channel response
            results = K.concatenate([y, c_hat], axis=-1)
            results = tf.expand_dims(results, axis=1)

            return results




