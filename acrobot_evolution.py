import gym
import numpy as np
import matplotlib.pyplot as plt
from random import choice
from schwimmbad import MultiPool

env = gym.make('Acrobot-v1')           # initialize environment

# each agents' weights are stored as a column in the weight matrix, randomly
# recombine weights and create new agents
def create_new_generation(pop_size, best_agents):
    num_params = best_agents.shape[1]
    new_pop = np.zeros([pop_size,num_params])

    '''                             randomly shuffle parameters from different good agents
    for a in range(pop_size):
        for p in range(num_params):
            new_pop[p,a] = np.random.normal(loc=choice(best_agents[p,:]), scale=0.3)
    '''
    for a in range(pop_size):
        new_pop[a,:] = [np.random.normal(loc=x, scale=0.3) for x in choice(best_agents)]

    return new_pop


# multiply observations with weights
def calc_action(observation, weights):
    dot_prod = observation.dot(weights)
    if dot_prod > .3:
        return 1
    elif dot_prod > -.3:
        return 0
    else:
        return -1


# run one game until pole falls or steps reaches 200
def run_episode(parameters, env=env):
    observation = env.reset()
    totalreward = 0
    for _ in range(1000):
        action = calc_action(observation, parameters)
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward


# run game once for all agents
def run_generation(env, parameters):
    num_agents = parameters.shape[0]
    rewards = np.zeros(num_agents)

    '''
    for a in range(num_agents):
        weights = parameters[a,:]
        rewards[a] = run_episode(env, weights)
    '''

    # replace loop with map operation for parallel processing
    with MultiPool() as pool:
        rewards = np.array(pool.map(run_episode, parameters))

    best_reward = np.max(rewards)
    mean_reward = np.mean(rewards)

    ranking = rewards.argsort()
    return parameters[ranking[-5:],:], best_reward, mean_reward, parameters[ranking[-1],:]


def run(env, num_episodes=200, pop_size=30):
    num_params = len(env.reset())
    parameters = np.random.rand(pop_size,num_params) * 2 - 1  # randomly initialize parameters
    best_rewards = np.zeros(num_episodes)
    mean_rewards = np.zeros(num_episodes)
    best_agent = parameters[0,:]

    for i in range(num_episodes):
        print("*** EPISODE ", i, " ***")
        # run simulation for every agent in population
        best_agents, best_reward, mean_reward, best_agent = run_generation(env, parameters)
        best_rewards[i] = best_reward
        mean_rewards[i] = mean_reward
        print("Best reward in generation: ", best_reward)
        print("Mean reward in generation: ", mean_reward)

        # generate new population
        parameters = create_new_generation(pop_size, best_agents)

    print("done training")
    return parameters, best_rewards, mean_rewards, best_agent

def plot_training(best_rewards, mean_rewards):
    plt.figure()
    plt.plot(best_rewards)
    plt.plot(mean_rewards)
    plt.legend(["best score", "mean score"])
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.show()

def play(env, weights):
    total_reward = 0
    observation = env.reset()
    done = False

    while not done:
        env.render()
        action = calc_action(observation, weights)
        observation, reward, done, info = env.step(action)
        total_reward += reward


weights, best_scores, mean_scores, best_agent = run(env, 200, 60)
print(weights)
plot_training(best_scores, mean_scores)
play(env, best_agent)
