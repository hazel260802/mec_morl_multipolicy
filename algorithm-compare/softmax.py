import numpy as np

# Define the softmax function
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def run_softmax(env, observation_space, num_episodes, alpha=0.1, gamma=0.99, tau=0.1):
    """
    Run the softmax policy gradient algorithm to learn a policy for the given environment.

    Args:
    - env: The environment to run the algorithm on.
    - observation_space: The observation space of the environment.
    - num_episodes: The number of episodes to run the algorithm for.
    - alpha: The learning rate (default is 0.1).
    - gamma: The discount factor (default is 0.99).
    - tau: The temperature parameter for the softmax function (default is 0.1).

    Returns:
    - avg_delays: A list containing the average delay per task for each episode.
    - avg_link_utilisations: A list containing the average link utilisation per task for each episode.
    """
    num_states = observation_space.shape[0]  # Number of states in the observation space
    num_actions = env.action_space.n  # Number of actions in the environment's action space
    Q = np.zeros((num_states, num_actions))  # Initialize the action-value function Q

    avg_delays = []  # List to store average delay per task for each episode
    avg_link_utilisations = []  # List to store average link utilisation per task for each episode
    
    for _ in range(num_episodes):
        state = env.reset()  # Reset the environment and get the initial state
        total_delay = 0  # Initialize total delay for this episode
        total_link_utilisation = 0  # Initialize total link utilisation for this episode
        total_tasks = 0  # Initialize total number of tasks for this episode
        done = False  # Initialize the done flag for the episode
        
        # Run the episode until termination
        while not done:
            # Calculate the mean values of each attribute across all servers
            column_sums = np.sum(state, axis=0)
            mean_values = column_sums / state.shape[0]
            
            # Calculate the final probabilities using the softmax function
            final_probabilities = softmax(mean_values)
            
            # Sample an action from the softmax distribution
            action = np.random.choice(num_actions, p=final_probabilities)
            
            # Take a step in the environment based on the selected action
            next_state, reward, done, _ = env.step(action)
            
            # Transition to the next state
            
            # Update the action-value function Q using the softmax policy gradient update
            if state.shape[0] < num_states and action < num_actions:
                Q[state.astype(int), action] += alpha * (reward + gamma * np.max(Q[next_state.astype(int)]) - Q[state.astype(int), action])
            
            # Transition to the next state
            state = next_state
        
        # Calculate average delay and link utilisation for this episode
        avg_delay, avg_link_utilisation = env.estimate_performance()
        
        avg_delays.append(avg_delay)  # Append the average delay per task for this episode to the list
        avg_link_utilisations.append(avg_link_utilisation)  # Append the average link utilisation per task for this episode to the list
    print("Average delay per task for each episode:", avg_delays)
    print("Average link utilisation per task for each episode:", avg_link_utilisations)
    return avg_delays, avg_link_utilisations
