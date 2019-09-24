import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from random import choice
from schwimmbad import MultiPool
from scipy.special import softmax

env = gym.make('Acrobot-v1')           # initialize environment

def calc_action(obs, params, actions):
    linear_activations = np.dot(params, obs)    # results in vector with activation for each action
    probs = softmax(linear_activations)         # probabilities with which each action is taken
    return np.random.choice(actions, p=probs)

def gradient():
    pass

def run_random(env, weights, max_iter = 1000):
    obs = env.reset()
    total_reward = 0
    actions = list()
    states = [obs]

    for i in range(max_iter):
        # generate new action
        actions.append(calc_action(observation, parameters))
        observation, reward, done, info = env.step(action)
        states.append(obs)
        totalreward += reward
        if done:
            break

    # done collecting data, calculate reward for each state action pair


def run():
    num_params = len(env.reset())
    actions = [-1,0,1]
    # initialize weights as tensor
    weights = torch.rand(len(actions), num_params, requires_grad=True)
