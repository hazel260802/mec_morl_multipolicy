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
    - avg_delays: A list containing the average delay per task for each episode.
    - avg_link_utilisations: A list containing the average link utilisation per task for each episode.
    """
    num_states = observation_space.shape[0]  # Number of states
    num_actions = env.action_space.n  # Number of actions
    Q = np.zeros((num_states, num_actions))  # Initialize Q-table

    avg_delays = []  # List to store average delay per task for each episode
    avg_link_utilisations = []  # List to store average link utilisation per task for each episode
    
    for _ in range(num_episodes):
        state = env.reset()  # Reset environment to initial state
        total_delay = 0  # Initialize total delay for this episode
        total_link_utilisation = 0  # Initialize total link utilisation for this episode
        total_tasks = 0  # Initialize total number of tasks for this episode
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

            if state.shape[0] in range(num_states) and action in range(num_actions):
                # Update Q-value using the Q-learning update rule
                Q[state.astype(int), action] += alpha * (reward + gamma * np.max(Q[next_state.astype(int)]) - Q[state.astype(int), action])
            
            state = next_state  # Move to the next state
        
        # Calculate average delay and link utilisation for this episode
        avg_delay, avg_link_utilisation = env.estimate_performance()
        
        avg_delays.append(avg_delay)  # Append the average delay per task for this episode to the list
        avg_link_utilisations.append(avg_link_utilisation)  # Append the average link utilisation per task for this episode to the list
    
    return avg_delays, avg_link_utilisations
