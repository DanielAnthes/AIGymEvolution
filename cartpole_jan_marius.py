import gym
import numpy as np

from CartpoleAgent import Agent

# Best agent ever: [-0.07033563  0.18317288  0.62339727  0.89762945]. Reward 500.0

env_max_episode_steps = 5000
env = gym.make('CartPole-v1')
env._max_episode_steps = env_max_episode_steps

population_size = 30
evolution_steps = 50
trials_per_agent = 20  # 50?
render = False

evolution_stats = []



def generate_evolution_step(agents, rewards, keep_stats=True):
    low_noise_sigma = 0.1
    high_noise_sigma = 0.5

    keep_n_best_agents = 5
    generate_n_new_agents = 5
    evolve_n_agents_low_noise = 5
    evolve_n_agents_high_noise = 5
    evolve_n_agents_from_best_low_noise = 5
    generate_n_rest = population_size - keep_n_best_agents - generate_n_new_agents - evolve_n_agents_low_noise - evolve_n_agents_high_noise - evolve_n_agents_from_best_low_noise

    new_agents = []

    best_n_agents = []
    for idx in rewards.argsort()[-keep_n_best_agents:]:
        agents[idx].age += 1
        best_n_agents.append(agents[idx])

    n_agents_low_noise = []
    for idx in range(evolve_n_agents_low_noise):
        old_weights = best_n_agents[idx].weights
        new_weights = old_weights + np.random.normal(0, low_noise_sigma, 4)
        n_agents_low_noise.append(Agent(weights=new_weights))

    n_agents_high_noise = []
    for idx in range(evolve_n_agents_high_noise):
        old_weights = best_n_agents[idx].weights
        new_weights = old_weights + np.random.normal(0, high_noise_sigma, 4)
        n_agents_high_noise.append(Agent(weights=new_weights))

    n_agents_from_best_low_noise = []
    old_weights = best_n_agents[0].weights
    for idx in range(evolve_n_agents_from_best_low_noise):
        new_weights = old_weights + np.random.normal(0, low_noise_sigma, 4)
        n_agents_from_best_low_noise.append(Agent(new_weights))

    new_agents.extend(best_n_agents)
    new_agents.extend([Agent() for _ in range(generate_n_new_agents)])
    new_agents.extend(n_agents_low_noise)
    new_agents.extend(n_agents_high_noise)
    new_agents.extend(n_agents_from_best_low_noise)
    new_agents.extend([Agent() for _ in range(generate_n_rest)])

    evolution_stats.append(rewards.argsort()[-keep_n_best_agents:])

    print(new_agents)
    print(evolution_stats)

    return new_agents

agents = [Agent() for _ in range(population_size)]

for evolution_step in range(evolution_steps):
    rewards = np.zeros(population_size)
    for agent_idx, agent in enumerate(agents):
        agent_trial_rewards = np.zeros(trials_per_agent)
        for agent_trial in range(trials_per_agent):
            env.reset()
            trial_reward = agent.play(env, render=render)
            agent_trial_rewards[agent_trial] = trial_reward
        agent_reward = np.median(agent_trial_rewards)
        print(f"Agent performance: {agent_reward}, weights: {agent}")
        rewards[agent_idx] = agent_reward
    agents = generate_evolution_step(agents, rewards)

