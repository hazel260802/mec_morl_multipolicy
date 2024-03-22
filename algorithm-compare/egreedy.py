import numpy as np

def run_egreedy(env, observation_space, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Runs the epsilon-greedy algorithm for reinforcement learning.

    Parameters:
    - env: The environment.
    - observation_space: The observation space of the environment.
    - num_episodes: Number of episodes to run the algorithm.
    - alpha: Learning rate.
    - gamma: Discount factor for future rewards.
    - epsilon: The probability of choosing a random action (exploration rate).

    Returns:
    - rewards: A list containing the total rewards for each episode.
    """
    num_states = observation_space.shape[0]  # Number of states
    num_actions = env.action_space.n  # Number of actions
    Q = np.zeros((num_states, num_actions))  # Initialize Q-table

    rewards = []  # List to store total rewards for each episode
    for _ in range(num_episodes):
        state = env.reset()  # Reset environment to initial state
        total_reward = 0  # Initialize total reward for this episode
        done = False  # Initialize flag indicating whether the episode has ended
        while not done:
            if np.random.rand() < epsilon:  # With probability epsilon, choose a random action
                action = env.action_space.sample()
            else:
                # Choose the action with the highest Q-value for the current state
                # If the state is out of bounds, choose a random action
                if np.all((0 <= state) & (state < num_states)):
                    max_action = np.argmax(Q[state.astype(int)])
                    if max_action in range(num_actions):
                        action = max_action
            next_state, reward, done, _ = env.step(action)  # Take a step in the environment
            total_reward += reward  # Accumulate the reward for this step
            if state.shape[0] in range(num_states) and action in range(num_actions):
                # Update Q-value using the Q-learning update rule
                Q[state.astype(int), action] += alpha * (reward + gamma * np.max(Q[next_state.astype(int)]) - Q[state.astype(int), action])
            state = next_state  # Move to the next state
        rewards.append(total_reward)  # Append the total reward for this episode to the list
    
    return rewards
