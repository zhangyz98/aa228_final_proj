from aa228_project_scenario import GoalFollowingScenario
import numpy as np
import gym
import time

if __name__ == "__main__":
    gfs = GoalFollowingScenario()
    #gfs = gym.make()
    dt = 0.1
    u = (0., 0.)
    for k in range(10000):
        state, reward, if_reset, non_defined = gfs.step(u) # move one time step and get the tuple of data
        gfs.render() # Test case
        # while time.time() - t < env.dt:
        #     pass
        # We should apply the reinforcement method here!
        # For example, policy = ReinforcementLearning() # (angular velocity, linear velocity)
        #              gfs.ego.set_control(policy)
        time.sleep(dt)