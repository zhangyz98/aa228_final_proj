from problem1_q_learning_env import *
import numpy as np
# added
import matplotlib.pyplot as plt

sim = simulator()

T = 5 * 365  # simulation duration
gamma = 0.95  # discount factor

# get historical data
data = generate_historical_data(sim)

# historical dataset:
# shape is 3*365 x 4
# k'th row contains (x_k, u_k, r_k, x_{k+1})

# TODO: write Q-learning to yield Q values,
# use Q values in policy (below)

num_iteration = 100
Q = np.zeros((len(sim.valid_states), np.max(sim.valid_actions) + 1))
Q_hist = np.zeros((len(sim.valid_states), np.max(sim.valid_actions) + 1, num_iteration * len(data)))

alpha = 0.01
for iteration in range(num_iteration):
    for d in range(len(data)):
        x_k, u_k, r_k, x_next = data[d]
        x_k, u_k, x_next = int(x_k), int(u_k), int(x_next)
        Q[x_k, u_k] += alpha * (r_k + gamma * np.max(Q[x_next, :]) - Q[x_k, u_k])
        Q_hist[:, :, iteration * len(data) + d] = Q

Q = np.delete(Q, obj=[1, 3], axis=1)
print('Qp=\n', Q)

#"""
for i in range(len(sim.valid_states)):
    plt.figure()
    for j in range(np.max(sim.valid_actions) + 1):
        if j % 2 != 0:
            continue
        plt.plot(np.arange(num_iteration * len(data)), Q_hist[i, j, :], '-.')
    plt.legend(['u=0', 'u=2', 'u=4'])
    plt.title('x=' + str(i))
    plt.xlabel('iteration')
    plt.ylabel('Q value')
    plt.savefig('p1_q_hist_x' + str(i))

def policy(state, Q):
    # TODO fill in
    return sim.valid_actions[np.argmax(Q[state, :])]


# Forward simulating the system
s = sim.reset()
r_hist = np.zeros(T)
r_total = 0
for t in range(T):
    a = policy(s, Q)
    sp, r = sim.step(a)
    s = sp
    # TODO add logging of rewards for plotting
    r_total += r
    r_hist[t] = r_total
plt.figure()
plt.plot(np.arange(T), r_hist)
plt.xlabel('day')
plt.ylabel('total profit')
plt.title('Profit Simulation')
plt.savefig('p1_total_profit')
plt.show()
#"""

# TODO: write value iteration to compute true Q values
# use functions:
# - sim.transition (dynamics)
# - sim.get_reward 
# plus sim.demand_probs for the probabilities associated with each demand value
eps = 1e-2
t = 0
Qv = np.zeros((len(sim.valid_states), np.max(sim.valid_actions)+1))
Q_prev = np.zeros((len(sim.valid_states), np.max(sim.valid_actions)+1)) + 1
while np.max(abs(Qv - Q_prev)) > eps:
    Q_prev[:, :] = Qv[:, :]
    for x in sim.valid_states:
        for u in sim.valid_actions:
            reward = 0
            reward_next = 0
            for i in range(len(sim.demand_probs)):
                reward += sim.demand_probs[i] * sim.get_reward(x, u, sim.valid_demands[i])
                x_next = sim.transition(x, u, sim.valid_demands[i])
                reward_next += gamma * sim.demand_probs[i] * np.max(Qv[x_next, :])
            Qv[x, u] = reward + reward_next

Qv = np.delete(Qv, obj=[1, 3], axis=1)
print('Qv=\n', Qv)



