import numpy as np

def run_egreedy(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space
    Q = np.zeros((num_states, num_actions))

    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                print("State:", state)  # Print state for debugging
                action = np.argmax(Q[state.astype(int)])
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            Q[state.astype(int), action] += alpha * (reward + gamma * np.max(Q[next_state.astype(int)]) - Q[state.astype(int), action])
            state = next_state
        rewards.append(total_reward)
    
    return rewards
