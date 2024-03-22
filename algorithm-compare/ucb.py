import numpy as np

def ucb1(Q, N, t, c=2):
    """
    UCB1 algorithm for action selection.
    
    Args:
    - Q: Array of action values.
    - N: Array of visit counts for each action.
    - t: Total number of time steps so far.
    - c: Exploration-exploitation trade-off parameter.
    
    Returns:
    - Action selected by UCB1 algorithm.
    """
    # Calculate UCB values for each action
    ucb_values = Q + c * np.sqrt(np.log(t) / (N + 1e-8))  # Adding a small value to prevent division by zero
    return np.argmax(ucb_values)  # Select action with the highest UCB value

def run_ucb1(env, observation_space, num_episodes, alpha=0.1, gamma=0.99, c=2):
    """
    Runs the UCB1 algorithm for action selection in a given environment.

    Args:
    - env: The environment.
    - observation_space: The observation space of the environment.
    - num_episodes: Number of episodes to run the algorithm.
    - alpha: Learning rate.
    - gamma: Discount factor for future rewards.
    - c: Exploration-exploitation trade-off parameter for UCB1.

    Returns:
    - avg_delays: A list containing the average delay per task for each episode.
    - avg_link_utilisations: A list containing the average link utilisation per task for each episode.
    """
    num_states = observation_space.shape[0]  # Number of states
    num_actions = env.action_space.n  # Number of actions
    Q = np.zeros((num_states, num_actions))  # Initialize action-value estimates
    N = np.zeros((num_states, num_actions))  # Count of visits for each action
    
    avg_delays = []  # List to store average delay per task for each episode
    avg_link_utilisations = []  # List to store average link utilisation per task for each episode
    
    for _ in range(num_episodes):
        state = env.reset()  # Reset the environment to initial state
        total_delay = 0  # Initialize total delay for this episode
        total_link_utilisation = 0  # Initialize total link utilisation for this episode
        total_tasks = 0  # Initialize total number of tasks for this episode
        done = False  # Initialize flag indicating whether the episode has ended
        t = 0  # Total time steps
        while not done:
            t += 1  # Increment time step
            if np.all((0 <= state) & (state < num_states)):
                max_action = ucb1(Q[state.astype(int)], N[state.astype(int)], t, c)  # Select action using UCB1
                if max_action in range(num_actions):
                            action = max_action
            next_state, reward, done, _ = env.step(action)  # Take a step in the environment
            # Update action-value estimates and visit counts using Q-learning update rule
            if state.shape[0] in range(num_states) and action in range(num_actions):
                N[state.astype(int), action] += 1
                Q[state.astype(int), action] += alpha * (reward + gamma * np.max(Q[next_state.astype(int)]) - Q[state.astype(int), action])
            state = next_state  # Move to the next state
        
        # Calculate average delay and link utilisation for this episode
        avg_delay, avg_link_utilisation = env.estimate_performance()
        
        avg_delays.append(avg_delay)  # Append the average delay per task for this episode to the list
        avg_link_utilisations.append(avg_link_utilisation)  # Append the average link utilisation per task for this episode to the list
    
    return avg_delays, avg_link_utilisations
