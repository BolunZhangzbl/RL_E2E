import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Activation, Input, Dense, GaussianNoise, Lambda, Dropout, Concatenate, Reshape, Add, Embedding, Flatten, Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from OUAction_Noise import OUActionNoise


class DDPGAgent():
    def __init__(self, env_, buffer_capacity=100000, batch_size=5120, critic_lr=0.0002, actor_lr=0.0001):
        self.env = env_

        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.shape

        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, 1, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions[0], self.num_actions[1]))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, 1, self.num_states))

        # Define the models
        self.actor_model = self.create_actor()
        self.target_actor = self.create_actor()
        self.target_actor.set_weights(self.actor_model.get_weights())

        self.critic_model = self.create_critic()
        self.target_critic = self.create_critic()
        self.target_critic.set_weights(self.critic_model.get_weights())

        # define optimizers
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        # define hyper-parameters
        self.gamma = 0.01
        self.tau = 0.001
        
        # define the noise object for action noise
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.02)*np.ones(1))

    def create_actor(self):
        actor_inp = Input((1, self.num_states))
        conv0 = Conv1D(self.num_states, kernel_size=1)(actor_inp)
        bn0 = BatchNormalization()(conv0)
        ac0 = Activation('elu')(bn0)

        conv1 = Conv1D(self.num_states, kernel_size=1)(ac0)
        bn1 = BatchNormalization()(conv1)
        ac1 = Activation('elu')(bn1)

        conv2 = Conv1D(self.num_actions[1], kernel_size=1)(ac1)
        bn2 = BatchNormalization()(conv2)
        ac2 = Activation('linear')(bn2)

        norm = Lambda(lambda x: x / K.sqrt(K.mean(x**2)))(ac2)

        return Model(actor_inp, norm)

    def create_critic(self):
        state_inp = Input((1, self.num_states))
        state_flat = Flatten()(state_inp)
        state_out = Dense(32, activation='elu')(state_flat)

        action_inp = Input((self.num_actions))
        action_flat = Flatten()(action_inp)
        action_out = Dense(32, activation='elu')(action_flat)

        concat = Concatenate()([state_out, action_out])

        d0 = Dense(256, activation='elu')(concat)
        bn_d0 = BatchNormalization()(d0)
        d1 = Dense(256, activation='elu')(bn_d0)
        bn_d1 = BatchNormalization()(d1)
        d2 = Dense(1, activation='linear')(bn_d1)
        bn_d2 = BatchNormalization()(d2)

        return Model([state_inp, action_inp], bn_d2)

    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic([next_state_batch, target_actions], training=True)
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

        # update the target_actor and the target_critic
        self.update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
        self.update_target(self.target_critic.variables, self.critic_model.variables, self.tau)

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        self.batch_indices = np.random.choice(record_range, self.batch_size)  # shape = (64,)
        # batch_indices = np.random.choice((record_range, self.batch_size))      # shape = (1,)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[self.batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[self.batch_indices])  # (4,2)
        reward_batch = tf.convert_to_tensor(self.reward_buffer[self.batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[self.batch_indices])

        # state_batch = tf.expand_dims(state_batch, axis=1)
        # next_state_batch = tf.expand_dims(next_state_batch, axis=1)

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

    @tf.function
    def update_target(self, target_weights, weights, tau_):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau_ + a * (2 - tau_))

    def sample(self):
        # Randomly sample indices
        # sample_indices = np.random.choice(sample_range, self.batch_size)   # sample_range: the largest indice to be sampled
        # batch_size: the number of indices to be sampled
        sample_indices = np.random.choice(min(self.buffer_counter, self.buffer_capacity), self.batch_size)

        state_sample = self.state_buffer[sample_indices]
        action_sample = self.action_buffer[sample_indices]
        # label_sample = np.array(self.label_buffer[sample_indices])

        return action_sample, state_sample

    def make_actions(self, states_tuple):
        #action = self.policy(state_tuple, self.ou_noise)
        action = self.actor_model(states_tuple)
        return action
        
    # policy() returns an action sampled from our Actor network plus some noise for exploration

    def policy(self, state, noise_object):
        sampled_actions = tf.squeeze(actor_model(state))
        noise = noise_object()
        scaling_factor = np.random.randn()

        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise * scaling_factor   # dynamic AWGN
        legal_action = sampled_actions
         # We make sure action is within bounds
         #legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
        return [np.squeeze(legal_action)]





