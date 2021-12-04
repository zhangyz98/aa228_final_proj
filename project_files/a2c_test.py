
import os
import numpy as np
from numpy import random
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
import logging
from scipy.special import softmax
from aa228_project_scenario import GoalFollowingScenario
from a2c_train import Actor

# suppress deprecation warning for now
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# maximum number of steps per episode
MAX_EP_STEPS = 200
# reward discount factor
GAMMA = .6
# once MAX_EPISODES or ctrl-c is pressed, number of test episodes to run
TEST_EPISODES = 5
# path where to save the actor after training
FROZEN_ACTOR_FILE = 'frozen_actor'

# setting the random seed makes things reproducible
random_seed = 2
np.random.seed(random_seed)
tf.compat.v1.random.set_random_seed(random_seed)
tf.keras.backend.set_floatx('float32')

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


def run_actor(env, actor, render=True):
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
    # Modification 10/19, 10/27
    env.T = 500*env.dt - env.dt/2. # Run for at most 200dt = 20 seconds

    # model = actor.buildnn()
    # # model.summary()
    # print(GRAPH_PB_PATH)
    # print(model.layers[0].get_weights())

    # model.load_weights(GRAPH_PB_PATH)
    # print(model.layers[0].get_weights())

    model = tf.keras.models.load_model(GRAPH_PB_PATH)

    for _ in range(TEST_EPISODES):
        # env.seed(int(np.random.rand()*1e6))
        env.seed(random_seed)
        obs, done = env.reset(), False
        total_reward = 0.

        while not done:
            t = time.time()
            # Modified 10/22
            # obs = np.array(obs).reshape(1,-1)
            obs = np.array(obs).reshape(-1)
            # print(obs, obs.shape)
            # print(obs[np.newaxis, :])
            predictions = model.predict(obs[np.newaxis, :])
            # print(predictions)
            action_probs = softmax(predictions)
            # print(action_probs)
            action_idx = np.random.choice(ACTION_DIM, p=action_probs[0, :])
            # action_idx = np.argmax(action_probs[0, :])
            # action_idx = actor.get_action(obs)
            action = all_action[action_idx]

            obs,reward,done,_ = env.step(action)

            total_reward += reward
            if render: # args.visualize:
                env.render()
                while time.time() - t < env.dt/2:
                    pass # runs 2x speed. This is not great, but time.sleep() causes problems with the interactive controller
            if render and done:
                env.close()
                if True: # args.visualize:
                    time.sleep(1)
                print("Reward: ", str(total_reward))

if __name__ == "__main__":
    GRAPH_PB_PATH = None

    with open("train.log", 'r') as file:
        lines = file.readlines()
        # print(lines)
        last_line = lines[-1]
        last_line = last_line[:-1]
        GRAPH_PB_PATH = './' + last_line + '/' + FROZEN_ACTOR_FILE
    print(GRAPH_PB_PATH)

    # with tf.compat.v1.Session() as sess:
    #     with tf.io.gfile.GFile(GRAPH_PB_PATH, 'rb') as f:
    #         graph_def = tf.compat.v1.GraphDef()
    #     graph_def.ParseFromString(f.read())
    #     sess.graph.as_default()
    #     tf.import_graph_def(graph_def, name='')
    #     graph_nodes = [n for n in graph_def.node]
    #     names = []
    #     for t in graph_nodes:
    #         names.append(t.name)
    #     print(names)

        # sess = tf.compat.v1.Session()

        # imported = tf.saved_model.load(GRAPH_PB_PATH)        

    env = GoalFollowingScenario()

    state_dim = env.observation_space.shape[0]
    action_dim = ACTION_DIM  # env.action_space.n

    # create an actor and a critic network and initialize their variables
    # output_layer = 'sequential/actor_outputs/BiasAdd'
    # input_node = 'actor_state_input'
    # prob_tensor = sess.graph.get_tensor_by_name(output_layer)
    # pred = sess.run(prob_tensor, {input_node: })

    sess = tf.compat.v1.Session()
    actor = Actor(sess, state_dim, action_dim)
    # actor.loadnn(GRAPH_PB_PATH)
    run_actor(env, actor, True)
