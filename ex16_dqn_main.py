import collections
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from stable_baselines3.common.vec_env import DummyVecEnv
from env import WareHouseEnv  # env 모듈 경로를 실제 모듈 위치로 변경
import numpy as np
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
# 경고메세지 끄기
warnings.filterwarnings(action='ignore')

# 하이퍼파라미터
learning_rate = 0.0005
gamma = 0.98
# 더 큰 리플레이 버퍼 크기로 시도
buffer_limit = 10000
batch_size = 32


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)  # Directly append the action without .item()
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return (
            torch.tensor(np.array(s_lst), dtype=torch.float),
            torch.tensor(np.array(a_lst), dtype=torch.long).view(-1, 1),
            torch.tensor(np.array(r_lst), dtype=torch.float),
            torch.tensor(np.array(s_prime_lst), dtype=torch.float),
            torch.tensor(np.array(done_mask_lst), dtype=torch.float).squeeze()
        )

    def size(self):
        return len(self.buffer)




class Qnet(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)  # Output layer for num_actions

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        obs = obs.view(1, -1)  # Flatten the observation and add batch dimension
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, out.size(-1) - 1)
        else:
            return out.argmax().item()

def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
 
        s = s.view(s.size(0), -1)  # Reshaping to [batch_size, -1]
        s_prime = s_prime.view(s_prime.size(0), -1)

        q_out = q(s)  # Q-values for all actions
        # Ensure that 'a' is in the correct shape for gather
        a = torch.tensor(a, dtype=torch.long).clone().detach().view(-1, 1)  # Convert 'a' to 2D tensor
        # Use gather to select the Q-values for the actions taken
        q_a = q_out.gather(1, a)  # This should align the dimensions correctly

        r = r.view(-1, 1)
        done_mask = done_mask.view(-1, 1)
        
        # Calculate the target Q-values
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask

        loss = F.mse_loss(q_a, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    num_lifts = 2
    num_robots = 3
    map_size = 4
    env = DummyVecEnv([lambda: WareHouseEnv(map_size=map_size, max_steps=5000, graphic=0, fps=150)])

    # Instantiate Qnet with the correct number of input features
    q_models = [Qnet(map_size**2, 4) for _ in range(num_lifts + num_robots)]
    q_targets = [Qnet(map_size**2, 4) for _ in range(num_lifts + num_robots)]

    for i in range(num_lifts + num_robots):
        q_targets[i].load_state_dict(q_models[i].state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    optimizers = [optim.Adam(q.parameters(), lr=learning_rate) for q in q_models]
    reward_return_list = []
    epsilon_return_list = []
    for n_epi in range(3000):
        # 다양한 엡실론 감쇠 일정 실험
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Adjusted epsilon decay
        s = env.reset()
        step_count = 0
        score = 0.0

        done = False

        while not done:
            a = []
            step_count += 1
            for i in range(num_lifts + num_robots):
                action = q_models[i].sample_action(torch.from_numpy(s).float(), epsilon)
                a.append(action)
                # Convert separate actions into a single action for the environment
            # 모든 에이전트의 액션을 하나의 정수로 결합
            combined_action = sum([a[i] * (4 ** i) for i in range(num_lifts + num_robots)])
            # 이 정수를 리스트로 변환하여 환경에 전달
            s_prime, r, done, _ = env.step([combined_action])

            # done = (terminated or truncated)
            done_mask = 0.0 if done else 1.0
            for i in range(num_lifts + num_robots):
                memory.put((s, a[i], r, s_prime, done_mask))  # Store experience for each agent
            s = s_prime

            score += r
            if done:
                print(f"Episode: {n_epi}, step: {step_count}, Reward: {score}")
                
                break

        if memory.size() > 30000: #경험 리플레이 메모리의 크기가 일정 이상이 되면, 즉 memory.size()가 50000보다 크면, 학습을 수행하는 부분
            for i in range(num_lifts + num_robots):
                train(q_models[i], q_targets[i], memory, optimizers[i])  # Train each model separately


        if n_epi % print_interval == 0 and n_epi != 0:
            for i in range(num_lifts + num_robots):
                q_targets[i].load_state_dict(q_models[i].state_dict())
            print("Episode: {}, Score: {:.1f}, Buffer: {}, Eps: {:.1f}%".format(
                n_epi, float(score / print_interval), memory.size(), float(epsilon * 100)
                ))
            reward_return_list.append(score/print_interval)
            epsilon_return_list.append(n_epi)
            score = 0.0

       
    env.close()
    now = datetime.now()
    nowtxt = now.strftime('%Y-%m-%d%H:%M:%S')
    # plt.plot(reward_return_list)
    # plt.xlabel('Iteration')
    # plt.ylabel('Reward Origin_DQN')
    # plt.savefig('Reward_Origin_DQN'+nowtxt+'.png', format='png', dpi=300)
    # # Display the plot
    # plt.show()

    plt.plot(epsilon_return_list)
    plt.xlabel('Iteration')
    plt.ylabel('Epsilon Origin_DQN')
    plt.savefig('Epsilon_Origin_DQN'+nowtxt+'.png', format='png', dpi=300)
    # Display the plot
    plt.show()

if __name__ == '__main__':
    main()
