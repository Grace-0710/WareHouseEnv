import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from env2 import WareHouseEnv
import math


# DQN 네트워크 정의
class DQN(nn.Module):
    def __init__(self, obs_space, action_space):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
        )

    def forward(self, x):
        return self.fc(x)

# 경험 리플레이 메모리
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN 에이전트
class DQNAgent:
    def __init__(self, obs_space, action_space, learning_rate=1e-4, gamma=0.99, memory_size=10000, batch_size=64):
        self.obs_space = obs_space
        self.action_space = action_space
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_size)
        self.model = DQN(obs_space, action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
            action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_space)
        return action

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# 환경 및 하이퍼파라미터 설정
env = WareHouseEnv(graphic=False)
obs_space = env.observation_space.shape[0] * env.observation_space.shape[1]
action_space = env.action_space.n

agent = DQNAgent(obs_space, action_space)
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

# 학습 루프
num_frames = 10000
for frame_idx in range(1, num_frames + 1):
    state = env.reset()
    state = np.reshape(state, obs_space)
    epsilon = epsilon_by_frame(frame_idx)
    total_reward = 0
    steps = 0

    while True:
        action = agent.select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, obs_space)
        
        agent.memory.push(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
        steps += 1
        
        agent.learn()
        
        if done:
            print(f"Episode finished after {steps} steps with soil sum {np.sum(env.cargo_map)}.")
            break

    print(f"Frame: {frame_idx}, Total Reward: {total_reward}, Epsilon: {epsilon}")

env.close()
