import gym
import numpy as np
import matplotlib.pyplot as plt
from random import choice
from schwimmbad import MultiPool
from scipy.special import softmax

env = gym.make('Acrobot-v1')           # initialize environment

def action(obs, params):
    linear_activations = np.dot(params, obs)    # results in vector with activation for each action
    probs = softmax(linear_activations)         # probabilities with which each action is taken
    return np.random.choice([-1,0,1], p=probs)
