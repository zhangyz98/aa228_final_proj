import gym
import os
import time
import random
import numpy as np
from collections import deque
import tensorflow as tf
import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from aa228_project_scenario import GoalFollowingScenario

# Set up environment
env = GoalFollowingScenario()

# Hyperparameters
state_size = 8 # This has to be from (env.observation_space)
action_size = 5 # This has to be from (env.action_space)
batch_size = 32 # batch size need to tune
n_episodes = 1001 # episodes need to tune
output_dir = "model_output/carlo_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

### Define agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000) # maximum batch numbers 2000

        self.gamma = 0.95 # Discount factor, will be tuned

        self.epsilon = 1.0 # initial epsilon = 1
        self.epsilon_decay = 0.995 # exploration rate decay, will be tuned
        self.epsilon_min = 0.01 # minimum exploration rate, will be tuned

        self.learning_rate = 0.001 # neural network learning rate, will be tuned
        
        self.model = self._build_model()
    
    def _build_model(self):

        # entire model structure need to tune

        model = Sequential()

        model.add(Dense(48, input_dim=self.state_size, activation="relu")) # First layer, will be tuned
        model.add(Dense(24, activation="relu")) # Second layer, will be tuned
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state): #epsilon greedy to explore while exploiting
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            # when the agent reach the destination/crash/exceed time, we just use the final reward function value
            target = reward
            if not done: # when not done, calculate the value function using the lookforward equation
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

agent = DQNAgent(state_size, action_size)

### interaction ###
done = False
# train for 1000 samples
for e in range(n_episodes):

    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0.

    # simulation for one time
    for time in range(2000):

        #env.render() # See the training process

        action_idx = agent.act(state)

        action = env.action_space[action_idx]

        next_state, reward, done, _ = env.step(action)

        next_state = np.reshape(next_state, [1, state_size])

        # at every time step, store the state action pair in memory
        agent.remember(state, action_idx, reward, next_state, done)

        state = next_state

        total_reward += reward

        if done:
            #env.close()
            print("episode: {", e , "}/{" , n_episodes , "}, cumulative reward: {" , total_reward , "}, e: {" , agent.epsilon, "}")
            break 
        
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    if e % 50 == 0:
        agent.save(output_dir + "weights_" + ":04d".format(e) + ".hdf5")
        
