import gym
import numpy as np
import matplotlib.pyplot as plt
from egreedy import run_egreedy
from softmax import run_softmax
from ucb import select_action_ucb
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.SDN_env import SDN_Env

config = 'multi-edge'
cloud_num = 1
edge_num = 1

def main():
    # Initialize the SDN environment
    env = SDN_Env(conf_file='../env/config1.json', conf_name=config, w=1.0, fc=4e9, fe=2e9, edge_num=edge_num, cloud_num=cloud_num)
    # Set hyperparameters
    num_episodes = 1000  

    # Run ε-Greedy algorithm
    egreedy_rewards = run_egreedy(env, num_episodes=num_episodes)

    # Run Softmax algorithm
    softmax_rewards = run_softmax(env, num_episodes=num_episodes)

    # Run UCB1 algorithm
    ucb_rewards = []
    Q = np.zeros((env.observation_space.shape[0], env.action_space.n))
    N = np.zeros((env.observation_space.shape[0], env.action_space.n))
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        t = 0  # Time step for UCB
        episode_rewards = 0
        while not done:
            action = select_action_ucb(Q, N, state, t)
            next_state, reward, done, _ = env.step(action)
            episode_rewards += reward
            N[state, action] += 1
            alpha_t = 0.1 / N[state, action]  # Learning rate
            sample = reward + 0.99 * np.max(Q[next_state]) - Q[state, action]
            Q[state, action] += alpha_t * sample
            state = next_state
            t += 1
        ucb_rewards.append(episode_rewards)

    # Plotting the results
    plt.plot(np.arange(num_episodes), egreedy_rewards, label='ε-Greedy')
    plt.plot(np.arange(num_episodes), softmax_rewards, label='Softmax')
    plt.plot(np.arange(num_episodes), ucb_rewards, label='UCB1')
    plt.xlabel('Episode')
    plt.ylabel('Average Total Reward')
    plt.title('Comparison of ε-Greedy, Softmax, and UCB1 Algorithms')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
