import numpy as np
from env import WareHouseEnv
from dqn import DQN  # Assuming your DQN class is defined in a file named dqn.py
import warnings

# 경고 메시지 무시 설정
warnings.filterwarnings('ignore', category=UserWarning, module='stable_baselines3.common.vec_env.base_vec_env')

import time

# 환경 생성
env = WareHouseEnv(map_size=4, max_steps=5000, graphic=0, fps=30)

# Learning rate schedule function
def learning_rate_schedule(progress_remaining):
    return 0.001 * progress_remaining

# DQN model initialization
dqn = DQN(state_size=env.observation_space.shape[0], action_size=env.action_space.n, learning_rate=0.001, discount_factor=0.99, epsilon_decay=0.995)

# Training parameters
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while True:
        # Select action using epsilon-greedy policy
        action = dqn.select_action(state)

        # Take action and observe the next state and reward
        next_state, reward, done, _ = env.step(action)

        # Update Q-network
        dqn.update_q_table(state, action, reward, next_state, done)

        # Update state and total reward
        state = next_state
        total_reward += reward

        if done:
            break

    # Print the total reward for the episode
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Close the environment
env.close()
