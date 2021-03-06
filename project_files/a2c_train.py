import os
import numpy as np
from numpy import random
from numpy.lib.histograms import _search_sorted_inclusive
from numpy.lib.npyio import save
import tensorflow as tf
import gym
# import gym_carlo
import matplotlib.pyplot as plt
import argparse
import time
from datetime import datetime
# from utils import *
from tqdm import tqdm
import cv2
import logging
from aa228_project_scenario import GoalFollowingScenario

# suppress deprecation warning for now
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# maximum number of training episodes
TRAIN_EPISODES = 100 # 1000
# maximum number of steps per episode
# CartPole-V0 has a maximum of 200 steps per episodes
MAX_EP_STEPS = 200
# reward discount factor
GAMMA = .6
# once MAX_EPISODES or ctrl-c is pressed, number of test episodes to run
TEST_EPISODES = 5
# batch size used for the training
BATCH_SIZE = 200  # 500
# maximum number of transitions stored in the replay buffer
MAX_REPLAY_BUFFER_SIZE = BATCH_SIZE * 10
# explore index that encourages exploration
DECAY_RATE = .9
# reward that is returned when the episode terminates early (i.e. the controller fails)
# FAILURE_REWARD = -200. # -10.
# make a timestamp for file saving
TIMESTAMP = str(datetime.now())

# path where to save the actor after training
# FROZEN_ACTOR_PATH = 'frozen_actor.pb'
FROZEN_ACTOR_PATH = 'frozen_actor'

# Modified
all_throttle = np.array((-.5, 0., .5, 1.)) # np.arange(start=-.5, stop=2, step=1)
all_steer = np.array((-.5, -.2, 0., .2, .5)) # np.arange(start=-1, stop=2, step=1)
ACTION_DIM = len(all_throttle) * len(all_steer)
all_action = np.zeros((ACTION_DIM, 2))
for i in range(len(all_steer)):
    for j in range(len(all_throttle)):
        all_action[i * len(all_throttle) + j][0] = all_steer[i]
        all_action[i * len(all_throttle) + j][1] = all_throttle[j]
print(f"action space = {all_action}, action_dim = {ACTION_DIM}")


# setting the random seed makes things reproducible
random_seed = 2
np.random.seed(random_seed)
tf.compat.v1.random.set_random_seed(random_seed)
tf.keras.backend.set_floatx('float32')


class Actor():
    def __init__(self, sess, state_dim, action_dim):
        """
        An actor for Actor-Critic reinforcement learning. This actor represent
        a stochastic policy. It predicts a distribution over actions condition
        on a given state. The distribution can then be sampled to produce
        an single control action.

        arguments:
            sess: a tensorflow session
            state_dim: an integer, number of states of the system
            action_dim: an integer, number of possible actions of the system
        returns:
            nothing
        """
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim

        # those placeholders serve as "inputs" to the computational graph.

        # state_input_ph is the input to the neural network
        self.state_input_ph = tf.compat.v1.placeholder(tf.float32, [None, state_dim], name='actor_state_input')
        
        # action_ph will be a label in the training process of the actor
        self.action_ph = tf.compat.v1.placeholder(tf.int32, [None, 1], name='actor_action')
        
        # td_error_ph will also be a label in the training process of the actor
        self.td_error_ph = tf.compat.v1.placeholder(tf.float32, [None, 1], name='actor_td_error')

        # setting up the computation graph

        # the neural network (input will be state, output is unscaled probability distribution)
        # note: the neural network must be entirely linear to support verification
        self.nn = self.buildnn()

        # probability distribution over potential actions
        self.action_probs = tf.math.softmax(self.nn(self.state_input_ph))
        # print("action_probs: ", self.action_probs)

        # convert action label to one_hot format
        self.action_one_hot = tf.one_hot(self.action_ph[:, 0], self.action_dim, dtype='float32')
        # print("action_ph: ", self.action_ph)
        # print("one_hot: ", self.action_one_hot)

        # log of the action probability, cliped for numerical stability
        # Modiifed 10/22
        self.log_action_prob = tf.reduce_sum(tf.math.log(
            tf.clip_by_value(self.action_probs, 1e-14, 1.)) * self.action_one_hot, axis=1, keepdims=True)
        # print("log_action_prob: ", self.log_action_prob)

        # the expected reward to go for this sample (J(theta)) (Eqn. 11)
        self.expected_v = self.log_action_prob * self.td_error_ph
        # taking the negative so that we effectively maximize
        self.loss = -tf.reduce_mean(self.expected_v)
        # the training step
        # Modified 12/2
        if TRAIN_EPISODES > 100:
            lr = .0001
        else:
            lr = .001
        # self.train_op = tf.compat.v1.train.AdamOptimizer(.001).minimize(self.loss)
        self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)
    
    def buildnn(self):
        model = tf.keras.Sequential([
            # Modified 10/27
            # tf.keras.layers.Dense(128, activation='relu',
            tf.keras.layers.Dense(64, activation='tanh',
                                  input_shape=[self.state_dim],
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(.1),
                                  name='actor_h1'),
            tf.keras.layers.Dense(128, activation='tanh',
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(.1),
                                  name='actor_h2'),
            tf.keras.layers.Dense(self.action_dim, activation=None,
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(.1),
                                  name='actor_outputs'),
        ])
        return model

    def savenn(self, path):
        # self.nn.save_weights(path)
        self.nn.save(path)
    
    def loadnn(self, path):
        self.nn.load_weights(path)

    def train_step(self, state, action, td_error):
        """
        Runs the training step

        arguments:
            state: a tensor representing a batch of states (batch_size X state_dim)
            action: a tensor of integers representing a batch of actions (batch_size X 1)
            where the integers correspond to the action number (0 indexed)
            td_error: a tensor of floats (batch_size X 1) the temporal differences
        returns:
            expected_v: a tensor of the expected reward for each of the samples
            in the batch (batch_size X 1)
        """
        expected_v, _ = self.sess.run([self.expected_v, self.train_op],
                                      {self.state_input_ph: state,
                                       self.action_ph: action,
                                       self.td_error_ph: td_error})

        # print("actor_action: ", action)
        # print("actor_state: ", state)
        # print("actor_td_error: ", td_error)
        # print("actor_expected_v: ", expected_v)
        return expected_v

    def get_action(self, state, explore_ind=0, train=False):
        """
        Get an action for a given state by predicting a probability distribution
        over actions and sampling one.

        arguments:
            state: a tensor of size (state_dim) representing the current state
        returns:
            action: an integer (0 indexed) corresponding to the action taken by the actor
        """
        action_probs = self.sess.run(self.action_probs,
                                     {self.state_input_ph: state[np.newaxis, :]})
        # print("action_probs: ", action_probs)
        # Modified 10/21
        # action = np.random.choice(self.action_dim, p=action_probs[0, :])
        # action_idx = np.random.choice(self.action_dim)
        # if train:
        #     if np.random.rand() >= explore_ind:
        #         action_idx = np.random.choice(self.action_dim, p=action_probs[0, :])
        action_idx = np.random.choice(self.action_dim, p=action_probs[0, :])
        # action = all_action[action_idx]
        # print("get_action result:", action_idx, action)
        return action_idx

    def export(self, frozen_actor_path):
        """
        Exports the neural network underlying the actor and its weights

        arguments:
            frozen_actor_path: a string, the path where to save the actor network
        returns:
            nothing
        """
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(self.sess,
                                                                                  tf.compat.v1.get_default_graph().as_graph_def(),
                                                                                  ['sequential/actor_outputs/BiasAdd'])
        with tf.io.gfile.GFile(frozen_actor_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())


class Critic():
    def __init__(self, sess, state_dim):
        """
        A critic for Actor-Critic reinforcement learning. This critic works
        by estimating a value function (expected reward-to-go) for given
        states. It is trained using Temporal Difference error learning (TD error).

        arguments:
            sess: tensorflow session
            state_dim: an interger, number of states of the system
        returns:
            nothing
        """
        self.sess = sess
        self.state_dim = state_dim

        # those placeholders serve as "inputs" to the computational graph.

        # state_input_ph is the input to the neural network
        self.state_input_ph = tf.compat.v1.placeholder(tf.float32, [None, state_dim], name='critic_state_input')
        # reward_ph will be a label during the training process
        self.reward_ph = tf.compat.v1.placeholder(tf.float32, [None, 1], name='critic_reward_input')
        # v_next will be a 'label' during the training process, even though it is 
        # produced by the nerual network as well
        self.v_next_ph = tf.compat.v1.placeholder(tf.float32, [None, 1])

        # setting up the computation graph

        ######### Your code starts here #########
        # hint: look at the implementation of the actor, the TD error and
        # the loss functions described in the writeup. An neural network architecture
        # identical to the one used by the actor should do the trick, but feel free to
        # experiment!
        # Note that the current train_step and train_op code expect you to compute three
        # member variables: self.v (the reward-to-go), self.loss and self.td_error

        # the neural network (input will be the current state, output is an 
        # estimate of the reward-to-go)

        self.nn = tf.keras.Sequential([
            # Modified 10/27
            # tf.keras.layers.Dense(128, activation='relu',
            tf.keras.layers.Dense(128, activation='tanh',
                                  input_shape=[state_dim],
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(.1),
                                  name='critic_h1'),
            # tf.keras.layers.Dense(128, activation='tanh',
            #                       kernel_initializer=tf.random_normal_initializer(0., .1),
            #                       bias_initializer=tf.constant_initializer(.1),
            #                       name='critic_h2'),
            tf.keras.layers.Dense(1, activation=None,
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(.1),
                                  name='critic_outputs'),
        ])

        # the estimate of reward-to-go (Eqn.12)
        self.v = self.nn(self.state_input_ph)
        # td_error (Eqn.14)
        self.td_error = self.reward_ph + GAMMA * self.v_next_ph - self.v

        # loss (Eqn.16)
        self.loss = tf.reduce_sum(tf.square(self.td_error))

        ######### Your code ends here #########

        # the train step
        self.train_op = tf.compat.v1.train.AdamOptimizer(.001).minimize(self.loss)

    def train_step(self, state, reward, state_next):
        """
        Runs the training step

        arguments:
            state: a tensor representing a batch of initial states (batch_size X state_dim)
            reward: a tensor representing a batch of rewards (batch_size X 1)
            state_next: a tensor representing a batch of 'future states' (batch_size X state_dim)
            each sample (state, reward, state_next) correspond to a given transition
        returns:
            td_error: the td errors of the batch, as a numpy array (batch_size X 1)
        """
        v_next = self.sess.run(self.v, {self.state_input_ph: state_next})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.state_input_ph: state,
                                     self.reward_ph: reward,
                                     self.v_next_ph: v_next})
        # print("critic_state_next = ", state_next)
        # print("critic_state = ", state)
        # print("critic_reward_ph = ", reward)
        # print("critic_v_next = ", v_next)
        # print("critic_td_error = ", td_error)
        # print("critic_td_error_mean = ", np.average(td_error))
        return td_error


def run_actor(env, actor, TRAIN_EPISODES, render=True, path=None):
    """
    Runs the actor on the environment for
    TRAIN_EPISODES

    arguments:
        env: the openai gym environment
        actor: an instance of the Actor class
        TRAIN_EPISODES: number of episodes to run the actor for
    returns:
        nothing
    """
    # Modification 10/19, 10/27, 12/2
    
    env.T = 500*env.dt - env.dt/2. # Run for at most 200dt = 20 seconds
    for test_eps in range(TEST_EPISODES):
        # env.seed(int(np.random.rand()*1e6))
        env.seed(random_seed)
        obs, done = env.reset(), False
        total_reward = 0.
        # if args.visualize:
        #     env.render()
        # frame = env.render(mode="rgb_array")
        # print(frame)
        # plt.imshow(frame)
        # frame_size = (frame.shape[0], frame.shape[1])
        # print(frame_size)
        # video_name = path + '/test' + str(test_eps) + '.avi'
        # video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), 10, frame_size)
        while not done:
            t = time.time()

            # Modified 10/22
            # obs = np.array(obs).reshape(1,-1)
            obs = np.array(obs).reshape(-1)
            # print(obs)
            action_idx = actor.get_action(obs)
            action = all_action[action_idx]
            obs,reward,done,_ = env.step(action)

            total_reward += reward
            if True: # args.visualize:
                env.render()
                # env_fig = env.render(mode="rgb_array")
                # print(env_fig)
                # video.write(env_fig)

                while time.time() - t < env.dt/2:
                    pass # runs 2x speed. This is not great, but time.sleep() causes problems with the interactive controller
            if done:
                env.close()
                if True: # args.visualize:
                    time.sleep(1)
                print("Reward: ", str(total_reward))

        # while not done:
        #     t = time.time()
        #     # obs = np.array(obs).reshape(1,-1)
        #     action = actor.get_action(obs)
        #     obs,_,done,_ = env.step(action)
        #     if args.visualize: 
        #         env.render()
        #         while time.time() - t < env.dt/2: pass # runs 2x speed. This is not great, but time.sleep() causes problems with the interactive controller
        #     if done:
        #         env.close()
        #         if args.visualize: time.sleep(1)
        #         if env.target_reached: success_counter += 1

    # for i_episode in range(TRAIN_EPISODES):
    #     state = env.reset()
    #     total_reward = 0.
    #     for t in range(MAX_EP_STEPS):
    #         if render:
    #             env.render()

    #         action = actor.get_action(state)
    #         state, reward, done, info = env.step(action)
    #         total_reward += reward

    #         if done:
    #             print("Reward: ", str(total_reward))
    #             break


def train_actor_critic(sess):
    # Modification 10/19

    # setup the OpenAI gym environment
    # env = gym.make('CartPole-v0')
    # env.seed(random_seed)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n

    # if args.goal.lower() == 'all':
    #     env = gym.make(scenario_name + 'Scenario-v0', goal=len(goals[scenario_name]))
    # else:
    #     env = gym.make(scenario_name + 'Scenario-v0', goal=np.argwhere(np.array(goals[scenario_name])==args.goal.lower())[0,0]) # hmm, unreadable
    # # Modified 10/27
    # env.seed(random_seed)
    env = GoalFollowingScenario()

    state_dim = env.observation_space.shape[0]
    print(f"state_dim: {state_dim}")
    action_dim = ACTION_DIM  # env.action_space.n

    # create an actor and a critic network and initialize their variables
    actor = Actor(sess, state_dim, action_dim)
    critic = Critic(sess, state_dim)
    sess.run(tf.compat.v1.global_variables_initializer())

    # the replay buffer will store observed transitions
    # Modified 10/22
    replay_buffer = np.zeros((0, 2 * state_dim + 2))

    # you can stop the training at any time using ctrl+c (the actor will
    # still be tested and its network exported for verification

    # allocate memory to keep track of episode rewards
    reward_hist = [] # np.zeros(TRAIN_EPISODES)

    # allocate memory to keep track of td errors and expected reward-to-go
    td_err_hist = [] # np.zeros(TRAIN_EPISODES)
    expected_v_hist = []

    # decay exploration
    explore_ind = 1.

    try:
        for train_eps in tqdm(range(TRAIN_EPISODES)):

            # very inneficient way of making sure the buffer isn't too full
            if replay_buffer.shape[0] > MAX_REPLAY_BUFFER_SIZE:
                replay_buffer = replay_buffer[-MAX_REPLAY_BUFFER_SIZE:, :]

            # reset the OpenAI gym environment to a random initial state for each episode
            state = env.reset()
            episode_reward = 0.
            timestamp = time.time()
            for t in range(MAX_EP_STEPS):
                # print("training %d" % t)
                # uses the actor to get an action at the current state
                # Modifed 10/22
                # action = actor.get_action(state)
                action_idx = actor.get_action(state, train=True)
                action = all_action[action_idx]
                # print(action)

                # call gym to get the next state and reward, given we are taking action at the current state
                state_next, reward, done, _ = env.step(action)
                if train_eps % 50 == 0:
                    env.render()
                    while time.time() - timestamp < env.dt/2:
                        pass # runs 2x speed. This is not great, but time.sleep() causes problems with the interactive controller
                if done:
                    env.close()
                    # if True: # args.visualize:
                    #     time.sleep(1)

                # done=True means either the cartpole failed OR we've reached the maximum number of episode steps
                # Modified 10/22
                # if done and t < (MAX_EP_STEPS - 1):
                #     reward = FAILURE_REWARD

                # accumulate the reward for this whole episode
                episode_reward += reward
                # store the observed transition in our replay buffer for training
                # Modified 10/22
                # replay_buffer = np.vstack((replay_buffer, np.hstack((state, action, reward, state_next))))
                replay_buffer = np.vstack((replay_buffer, np.hstack((state, action_idx, reward, state_next))))

                # if our replay buffer has accumulated enough samples, we start learning the actor and the critic
                if replay_buffer.shape[0] >= BATCH_SIZE:

                    # we sample BATCH_SIZE transition from our replay buffer
                    samples_i = np.random.choice(replay_buffer.shape[0], BATCH_SIZE, replace=False)
                    # print("samples = ", samples_i)
                    state_samples = replay_buffer[samples_i, 0:state_dim]

                    # Modified 10/22
                    action_samples = replay_buffer[samples_i, state_dim:state_dim + 1]
                    reward_samples = replay_buffer[samples_i, state_dim + 1:state_dim + 2]
                    state_next_samples = replay_buffer[samples_i, state_dim + 2:2 * state_dim + 2]

                    # compute the TD error using the critic
                    td_error = critic.train_step(state_samples, reward_samples, state_next_samples)
                    td_err_hist.append(np.average(td_error))

                    # train the actor (we don't need the expected value unless you want to log it)
                    expected_v = actor.train_step(state_samples, action_samples, td_error)
                    expected_v_hist.append(np.average(expected_v))

                # update current state for next iteration
                state = state_next

                if done:
                    break
            # reward_hist[i_episode] = episode_reward
            reward_hist.append(episode_reward)

            if train_eps % 10 == 0:
                explore_ind *= DECAY_RATE


    except KeyboardInterrupt:
        print("training interrupted")
    
    # save trained actor
    # Modified 12/2
    savepath = './' + TIMESTAMP
    figpath = savepath + '/figs'
    videopath = savepath + '/videos'

    # for path in [savepath, figpath, videopath]:
    for path in [savepath, figpath]:
        if not os.path.exists(path):
            os.mkdir(path)
    
    plt.figure()
    plt.plot(reward_hist, '-.')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Episode Reward History')
    plt.savefig(figpath + '/reward_hist.png')
    plt.show()

    plt.figure()
    plt.plot(td_err_hist, '-.')
    plt.xlabel('Episode')
    plt.ylabel('Mean TD Error')
    plt.title('TD Error History')
    plt.savefig(figpath + '/td_err_hist.png')
    plt.show()

    plt.figure()
    plt.plot(expected_v_hist, '-.')
    plt.xlabel('Episode')
    plt.ylabel('Mean Expected Reward-To-Go')
    plt.title('Expected Reward-To-Go History')
    plt.savefig(figpath + '/expected_v_hist.png')
    plt.show()

    # exports the actor neural network and its weights, for future verification
    # actor.export(savepath + '/' + FROZEN_ACTOR_PATH)
    # actor.savenn(savepath + '/' + FROZEN_ACTOR_PATH + '.hdf5')
    actor.savenn(savepath + '/' + FROZEN_ACTOR_PATH)
    

    # run some test cases
    run_actor(env, actor, TEST_EPISODES, True, videopath)

    # closes the environement
    env.close()


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--scenario', type=str, help="intersection, circularroad, lanechange", default="intersection")
    # parser.add_argument('--goal', type=str, help="left, straight, right, inner, outer, all", default="all")
    # parser.add_argument("--visualize", action="store_true", default=False)
    # args = parser.parse_args()
    # scenario_name = args.scenario.lower()
    # assert scenario_name in scenario_names, '--scenario argument is invalid!'
    
    logging.basicConfig(
        filename='train.log',
        filemode='a',
        format='\n%(message)s',
        level=logging.INFO
        )
    logging.info(str(TIMESTAMP))

    sess = tf.compat.v1.Session()

    train_actor_critic(sess)

