import gym
import numpy as np
import matplotlib.pyplot as plt
from random import choice

env = gym.make('CartPole-v0')           # initialize environment
num_episodes = 20                       # number of episodes
pop_size = 30
weights = np.random.rand(4,pop_size) * 2 - 1  # randomly initialize parameters

# each agents' weights are stored as a column in the weight matrix, randomly
# recombine weights and create new agents
def create_new_generation(pop_size, best_agents):
    num_params = best_agents.shape[0]
    new_pop = np.zeros(num_params, pop_size)

    for a in range(pop_size):
        for p in num_params:
            new_pop[p,a] = choice(best_agents[p,:])

    return new_pop


# multiply observations with weights
def calc_action(observation, weights):
    return 1 if observation.dot(weights) > 0 else 0


# run one game until pole falls or steps reaches 200
def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        action = calc_action(parameters, weights)
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward


# run game once for all agents
def run_generation(env, parameters):
    num_agents = parameters.shape[1]
    rewards = np.zeros(num_agents)

    for a in range(num_agents):
        weights = parameters[:,a]
        rewards[a] = run_episode(env, weights)

    best_reward = np.max(rewards)
    mean_reward = np.mean(rewards)
    print("Best reward in generation: ", best_reward)
    print("Mean reward in generation: ", mean_reward)

    ranking = rewards.argsort()
    best_weights = parameters[:,ranking[-5:]] # matrix with 5 best agents


# run some episodes with random weights to test
rewards = np.zeros(num_episodes)
for i in range(num_episodes):
    print("*** EPISODE ", i, " ***")
    rewards[i] = run_episode(env, weights)

print(rewards)

plt.figure()
plt.plot(rewards)
plt.xlab("episode")
plt.ylab("reward")
plt.show()
