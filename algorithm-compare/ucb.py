import gym
import numpy as np
import math

# Define the UCB1 action selection function
def select_action_ucb(Q, N, state, t):
    num_actions = Q.shape[1]
    ucb_values = np.zeros(num_actions)
    for a in range(num_actions):
        if N[state, a] == 0:
            ucb_values[a] = float('inf')  # Assign infinity for unexplored actions
        else:
            q = Q[state, a]
            ucb_values[a] = q + 2 * math.log(t) / N[state, a]
    return np.argmax(ucb_values)

# Create the environment
env = gym.make('SDN_Env')

# Set hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
num_episodes = 1000  # Number of training episodes

# Initialize the Q-table and N-table
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))
N = np.zeros((num_states, num_actions))

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    t = 0  # Time step for UCB
    while not done:
        # Choose action using UCB1 algorithm
        action = select_action_ucb(Q, N, state, episode)
        
        # Take action and observe the next state and reward from the environment
        next_state, reward, done, _ = env.step(action)
        
        # Update the Q-table
        N[state, action] += 1
        alpha_t = alpha / N[state, action]  # Update learning rate
        sample = reward + gamma * np.max(Q[next_state]) - Q[state, action]
        Q[state, action] += alpha_t * sample
        
        # Update the current state
        state = next_state
        t += 1  # Increment time step
    
# Testing the trained agent
total_rewards = []
for _ in range(100):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = np.argmax(Q[state])  # Select the greedy action
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    total_rewards.append(total_reward)

# Print the average total reward
print("Average total reward:", np.mean(total_rewards))
