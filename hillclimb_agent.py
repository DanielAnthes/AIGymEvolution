import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

def calc_action(observation, weights):
    return 1 if observation.dot(weights) > 0 else 0

# run one game until pole falls or steps reaches 200
def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        action = calc_action(observation, parameters)
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

def run(env, num_episodes=200):
    parameters = np.random.rand(4) * 2 - 1  # randomly initialize parameters
    best_rewards = np.zeros(num_episodes)
    mean_rewards = np.zeros(num_episodes)

    # generate initial reward
    rewards = np.zeros(num_episodes)
    rewards[0] = run_episode(env, parameters)

    for i in range(1, num_episodes):
        print("*** EPISODE ", i, " ***")
        new_parameters = np.array([np.random.normal(x, 0.1) for x in parameters])

        # run episode with new parameters
        rewards[i] = run_episode(env, new_parameters)
        if rewards[i] >= rewards[i - 1]:
            parameters = new_parameters


    print("done training")
    return parameters, rewards

parameters, rewards = run(env)

plt.figure()
plt.plot(rewards)
plt.xlabel("episode")
plt.ylabel("reward")
plt.show()
