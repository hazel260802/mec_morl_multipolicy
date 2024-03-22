import gym
import numpy as np
import matplotlib.pyplot as plt
from egreedy import run_egreedy
from softmax import run_softmax
from ucb import run_ucb1
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
    
    observation_space = env.get_obs()
    # Run ε-Greedy algorithm
    egreedy_rewards = run_egreedy(env, observation_space, num_episodes=num_episodes)

    # Run Softmax algorithm
    softmax_rewards = run_softmax(env, observation_space, num_episodes=num_episodes)

    # Run UCB1 algorithm
    ucb_rewards = run_ucb1(env, observation_space, num_episodes=num_episodes)
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
