import numpy as np

def run_softmax(env, num_episodes=1000, alpha=0.1, gamma=0.99, tau=0.1):
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space
    Q = np.zeros((num_states, num_actions))

    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            logits = Q[state] / tau
            probabilities = np.exp(logits) / np.sum(np.exp(logits))
            action = np.random.choice(num_actions, p=probabilities)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
        rewards.append(total_reward)
    
    return rewards
