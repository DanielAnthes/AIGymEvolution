import numpy as np


class Agent:
    def __init__(self, weights=None):
        self.age = 0
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.rand(4) * 2 - 1

    def __str__(self):
        return f"Age: {self.age}, {self.weights}"

    def get_action(self, state):
        # Action 0: Push cart to the left
        # Action 1: Push cart to the right
        action = np.sum(self.weights * state)
        return (action > 0) * 1

    def play(self, env, render=False):
        state = env.state
        reward = 0
        while True:
            action = self.get_action(state)
            state, step_reward, done, _ = env.step(action)
            if render:
                env.render()
            reward += step_reward
            if done:
                # print(f"Simulation of agent {agent} finished. Reward {reward}")
                break
        return reward
