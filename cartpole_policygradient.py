import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from random import choice

# initialize environment
env = gym.make('CartPole-v1')
num_rnd = 100

# initialize memory
observations = list()
actions = list()
rewards = list()

NUM_OBSERVATIONS = 4
ACTIONS = [0,1]
NUM_ACTIONS = len(ACTIONS)

weights = torch.zeros(NUM_ACTIONS,NUM_OBSERVATIONS, requires_grad=True)
softmax = torch.nn.Softmax(dim=0)

def take_action(weights, observation):
    print(weights)
    print(torch.mm(weights,torch.tensor(observation)))


    probs = softmax(torch.mm(weights,torch.tensor(observation)))
    return np.random.choice(ACTIONS, p=probs)


# random episodes for training
# TODO introduce discount
def episode(weights):
    observations = list()
    observations.append(env.reset())
    actions = list()
    rewards = list()
    for t in range(500):
        action = take_action(weights, observations[t])
        actions.append(action)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        observations.append(observation)
        if done:
            break
    # compute reward from every state onwards
    del observations[-1]

    # compute future rewards
    for i in range(len(rewards)):
        rewards[i] = sum(rewards[i:])

    return observations, actions, rewards


# play a number of random episodes to gather training data
for _ in range(num_rnd):
    o,a,r = episode(weights)
    observations += o
    actions += a
    rewards += r
