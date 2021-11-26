from aa228_project_scenario import GoalFollowingScenario
from aa228_interset_scenario import IntersectionScenario
import numpy as np
import gym
import time

if __name__ == "__main__":
    gfs = IntersectionScenario()
    #gfs = gym.make()
    dt = 0.1
    u = (0., 0.)
    total_reward = 0.
    while True:
        state, reward, if_reset, non_defined = gfs.step(u) # move one time step and get the tuple of data
        gfs.render() # Test case
        # while time.time() - t < env.dt:
        #     pass
        # We should apply the reinforcement method here!
        # For example, policy = ReinforcementLearning() # (angular velocity, linear velocity)
        #              gfs.ego.set_control(policy)
        total_reward += reward
        if if_reset:
            gfs.close()
            gfs.reset()
            print("The total reward for this episode is: ", total_reward)
            total_reward = 0.
        time.sleep(dt/4)