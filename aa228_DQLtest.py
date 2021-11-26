import gym
import os
import time
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from aa228_project_scenario import GoalFollowingScenario
from DQL import DQNAgent

# Set up environment
# env = ObstacleAvoidanceScenario()
env = GoalFollowingScenario()
state_size = 6
action_size = 3
dt = 0.2

agent = DQNAgent(state_size, action_size)
agent.load("model_output/carlo_modelweights_3000.hdf5")

done = False
for e in range(10):

    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0.

    for t in range(5000):

        env.render() # See the training process

        act_values = agent.model.predict(state)

        action_idx = np.argmax(act_values[0])

        action = env.action_space[action_idx]

        next_state, reward, done, _ = env.step(action)

        next_state = np.reshape(next_state, [1, state_size])

        state = next_state

        total_reward += reward

        time.sleep(dt/4)

        if done:
            env.close()
            print("attempt: {", e , "}/{cumulative reward: {" , total_reward , "}")
            break