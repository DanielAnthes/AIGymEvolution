import gym
import numpy as np

from CartpoleAgent import Agent

env_max_episode_steps = 15000
env = gym.make('CartPole-v1')
env._max_episode_steps = env_max_episode_steps

env.reset()

best_agent_ever = Agent(weights=np.array([0.32263838, 0.62688226, 0.97813645, 0.90945963]))

reward = best_agent_ever.play(env, render=False)

print(reward)

env.close()