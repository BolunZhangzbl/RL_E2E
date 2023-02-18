import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
############## Import SubClass ################

from Env import E2E_Env
from DDPG import DDPGAgent


##########################################################################
################## Define the Hyper-parameters ###########################
##########################################################################

msg_total = 256
k = int(np.log2(msg_total))
channel = 16
EbNodB_train = 20

BS = 5120

# define the channel parameters
EbNodB_train = 10.0 ** (EbNodB_train / 10.0)
R = k / float(channel)
noise_std = np.sqrt(1 / (2 * R * EbNodB_train))
#noise_std = np.sqrt(1 / (2 * EbNodB_train))   # noise_std = 0.0707 for channel=8, msg=256, EbNodB=20,
                                                  # noise_std = 0.2236 for chennel=8, msg=256, EbNodB=10


# define the objects
env = E2E_Env(msg_total=msg_total, channel=channel)
agent = DDPGAgent(env, 20000, BS)

# define the training parameters
total_episodes = 20000
max_step = 500

# To store history of each episode
ep_reward_list = []
# To store reward of each step (5000 episodes * 500 steps = 2500000 iterations)
ep_mean_reward_list = []
# To store average awardhistory of last few episodes
avg_reward_list = []

# To store the receiver loss and accuracy
loss_rx, acc_rx = [], []

try:
    # Take about 4 min to train
    for ep in range(total_episodes):

        prev_state = env.reset()
        episodic_reward = 0

        # while True:
        for t in range(max_step):

            # env.render()

            tf_prev_state = tf.convert_to_tensor(prev_state)

            # action = policy(tf_prev_state, ou_noise)
            action = agent.make_actions(tf_prev_state)
            # action = np.array(action)
            print(action)

            # Receive state and reward from environment
            state, reward, done = env.step(action, prev_state)  # state is next_state here
            print("Step/Episode: {}/{},   Reward: {}".format(t, ep, reward))
            # print("current state: {}, label: {}".format(prev_state, label))
            ep_mean_reward_list.append(reward)

            agent.record((prev_state, action, reward, state))  # prev_state != state (real problem!!!)
            episodic_reward += reward

            # Update the transmitter
            agent.learn()
            #update_target(target_actor.variables, actor_model.variables, tau)
            #update_target(target_critic.variables, critic_model.variables, tau)

            # Update the receiver
            # if len(ep_mean_reward_list)>1024 and len(ep_mean_reward_list)%2 == 0:
            if len(ep_mean_reward_list) % 2 == 0:
                data, label = agent.sample()
                sig_rx = env.cha_noise(data, noise_std)
                loss, acc = env.train_rx(sig_rx, label)
                loss_rx += loss
                acc_rx += acc
            else:
                pass

            # End this episode when done == True
            if done:
                print("\nEpisode: {},   Acc. Mean. Reward: {}".format(ep, np.mean(ep_mean_reward_list)))
                break

            prev_state = state  # current_state = next_state

        ep_reward_list.append(episodic_reward)  # accumulated reward for each episode
        # avg_reward_list.append(np.mean(ep_mean_reward_list)) # mean reward for each episode

        # Mean of last "n" episodes
        n = 50
        avg_reward = np.mean(ep_reward_list[-n:])
        avg_reward_list.append(avg_reward)  # Averaged reward of the last n episodes
        print("Episode * {} * Avg Reward is ==> {}\n".format(ep, avg_reward))

        if np.mean(avg_reward_list[-2000:]) >= -0.01 and np.mean(loss_rx[-1000:]) <= 0.01 and len(loss_rx) > 1000:
            break

finally:
    pass

    # Plotting Avg. Reward vs. Acc. Reward
    '''plt.plot(ep_reward_list, linewidth=2, label='Acc.')
    plt.plot(avg_reward_list, linewidth=2, label='Avg.')
    plt.title("Acc. vs Avg. Episodic Reward", fontsize=16)
    plt.xlabel("Episode", fontsize=15)
    plt.ylabel("Episodic Reward", fontsize=15)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid()
    plt.savefig("Episodic reward rayleigh msg=" + str(msg_total) + ".png")
    files.download("Episodic reward rayleigh msg=" + str(msg_total) + ".png")
    plt.show()
  
    # Plotting the Step reward for <=2500000 iterations
    plt.plot(ep_mean_reward_list, linewidth=2, label='DDPG')
    plt.title("Reward for each iteration", fontsize=16)
    plt.xlabel("Iteration", fontsize=15)
    plt.ylabel("Iterative Reward", fontsize=15)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid()
    plt.savefig("Iterative reward rayleigh msg=" + str(msg_total) + ".png")
    files.download("Iterative reward rayleigh msg=" + str(msg_total) + ".png")
    plt.show()
  
    # Plotting the receiver loss
    plt.plot(loss_rx, linewidth=2, label='Receiver')
    plt.title("Receiver Loss", fontsize=16)
    plt.xlabel("Iteration", fontsize=15)
    plt.ylabel("Iterative Loss (Receiver)", fontsize=15)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid()
    plt.savefig("Receiver loss rayleigh msg=" + str(msg_total) + ".png")
    files.download("Receiver loss rayleigh msg=" + str(msg_total) + ".png")
    plt.show()'''


#################################  Define Functions  #################################

# Plot BLER curve
def Call_BLER(channel_model, bertest_datasize=50000, EbNodB_low=0, EbNodB_high=20, EbNodB_num=41):
    test_data = np.random.randint(0, msg_total, bertest_datasize)  # Embedding data
    test_label = np.zeros((bertest_datasize, msg_total))
    test_label[np.arange(bertest_datasize), test_data] = 1  # Convert to one-hot vector
    test_cnn = np.expand_dims(test_label, axis=1)

    EbNodB_range = list(np.linspace(EbNodB_low, EbNodB_high, EbNodB_num))
    ber = [None] * len(EbNodB_range)

    for n in range(0, len(EbNodB_range)):
        EbNo = 10 ** (EbNodB_range[n] / 10.0)
        noise_std = np.sqrt(1 / (2 * R * EbNo))

        if channel_model == 'Rayleigh':
            encoded_signal = agent.make_actions(test_cnn)
            final_signal = env.make_noise(enc_tuple=encoded_signal, sigma=noise_std)
            # final_signal = channel_layer(x=encoded_signal, sigma=noise_std)

        if channel_model == 'AWGN':
            encoded_signal = agent.make_actions(test_cnn)
            final_signal = env.make_noise(enc_tuple=encoded_signal, sigma=noise_std)
            # final_signal = encoded_signal + noise

        pred_final_signal = env.predict_rx(final_signal)
        pred_final_signal = np.squeeze(pred_final_signal)
        pred_output = np.argmax(pred_final_signal, axis=1)  # axis=1 -> Convert one-hot vector to original data
        no_errors = (pred_output != test_data)
        no_errors = no_errors.astype(int).sum()
        ber[n] = no_errors / bertest_datasize
        print('SNR: ', EbNodB_range[n], 'BER: ', ber[n])

    return ber
  
  
