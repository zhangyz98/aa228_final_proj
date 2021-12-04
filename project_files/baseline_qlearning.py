import gym
import os
import time
import random
import numpy as np
from collections import deque
from tqdm import tqdm
from aa228_project_scenario import GoalFollowingScenario
from matplotlib import pyplot as plt
# Set up environment
env = GoalFollowingScenario()
MAX_EP_STEPS = 200
GAMMA = .6
alpha = 1.
explore_param = 1

output_dir = "./baseline_q/"

# Hyperparameters
state_size = 5 # 8 # This has to be from (env.observation_space)
action_size = 5 # This has to be from (env.action_space)
train_episodes = 1000 # episodes need to tune
test_episodes = 5

x_states = np.linspace(30, 50, 10)
y_states = np.linspace(0, 80, 20)
vel_states = np.linspace(0, 3, 3)
steer_states = np.linspace(-np.pi/2, np.pi/2, 3)
head_states = np.linspace(0, 2 * np.pi, 4)
target_states = np.array([0, 1])

# all_states = [x_states, y_states, vel_states, steer_states, head_states, target_states, target_states, target_states]
all_states = [x_states, y_states, target_states, target_states, target_states]

all_actions = np.array([(0, 1), (0.4, 0), (-0.4, 0), (0, -1), (0, 0)])


Q = np.zeros((
    len(x_states), len(y_states),
    # len(vel_states), len(steer_states), len(head_states),
    len(target_states), len(target_states), len(target_states),
    len(all_actions)
    ))
print(Q.shape)

def assigntostate(state, states=all_states):
    assigns = np.zeros(state_size)
    for i in range(state_size):
        assigns[i] = np.argmin(abs(state[i] - states[i]))
    return assigns

eps_errs = []
eps_rewards = []
for train_eps in tqdm(range(train_episodes)):
    state = env.reset()
    done = False
    eps_err = 0
    eps_reward = 0
    # decay the step size
    if train_eps % 100 == 0 and train_eps > 0:
        alpha *= .5
    if train_eps % 20 == 0 and train_eps > 0:
        explore_param *= .9

    for t in range(MAX_EP_STEPS):
        state = np.array(state).reshape(-1)

        state = [state[s] for s in [0, 1, 5, 6, 7]]
        state = np.array(state).reshape(-1)

        assigns = assigntostate(state, all_states)
        assigns = assigns.astype('int32')
        
        Q_assign = Q[
            assigns[0], assigns[1],
            assigns[2], assigns[3], assigns[4],
            # assigns[5], assigns[6], assigns[7],
            :]
        
        if np.random.rand() > explore_param:
            action_idx = np.argmax(Q_assign)
        else:
            action_idx = np.random.randint(action_size)
        action = all_actions[action_idx]

        next_state, reward, done, _ = env.step(action)
        eps_reward += reward

        idx = tuple(np.append(assigns, action_idx))
        update = alpha * (reward + GAMMA * np.max(Q_assign) - Q[idx])
        if train_eps % 10 == 0 and t % 10 == 0:
            print(f"update = {update}")
        Q[idx] += update
        eps_err = max(abs(update), abs(eps_err))

        if done:
            env.close()
            break
        
        state = next_state

    eps_errs.append(eps_err)
    eps_rewards.append(eps_reward)


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.figure()
plt.plot(eps_errs)
plt.savefig(output_dir + 'eps_err.png')
plt.show()

plt.figure()
plt.plot(eps_rewards)
plt.savefig(output_dir + 'eps_reward.png')


for test_eps in range(test_episodes):
    state = env.reset()
    for _ in range(MAX_EP_STEPS):
        t = time.time()

        state = np.array(state).reshape(-1)
        assigns = assigntostate(state, all_states)
        assigns = assigns.astype('int32')
        
        Q_assign = Q[
            assigns[0], assigns[1],
            assigns[2], assigns[3], assigns[4],
            # assigns[5], assigns[6], assigns[7],
            :]

        action_idx = np.argmax(Q_assign)
        action = all_actions[action_idx]

        next_state, reward, done, _ = env.step(action)

        env.render()
        while time.time() - t < env.dt / 2:
            pass
        if done:
            env.close()
            time.sleep(1)
            break


