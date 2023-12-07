# libraries
import gymnasium as gym
import collections
import random
import numpy as np

# pytorch library is used for deep learning
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

# hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000        # size of replay buffer
batch_size = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)    # double-ended queue

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            result_list = [[x] for x in a ]
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
            #print("result_list:::::",result_list)
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self,num_inputs, num_actions):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else :
            return out.argmax().item()

def train(q, q_target, memory, optimizer):
    for i in range(4):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        print("q_out::::",q_out)
       
        flattened_a = a.reshape(-1, q_out.shape[-1])
        print("flattened_a::::",flattened_a)
        q_a = q_out.gather(1,flattened_a)

        # DQN
        #max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)

        # Double DQN
        argmax_Q = q(s_prime).max(1)[1].unsqueeze(1)
        max_q_prime = q_target(s_prime).gather(1, argmax_Q)

        target = r + gamma * max_q_prime * done_mask

        # MSE Loss
        loss = F.mse_loss(q_a, target)

        # Smooth L1 Loss
        #loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    num_lifts = 2
    num_robots = 3
    map_size = 4
    env = DummyVecEnv([lambda: WareHouseEnv(map_size=map_size, max_steps=5000, graphic=0, fps=150)])

    q = [Qnet(map_size**2, 4) for _ in range(num_lifts + num_robots)]
    q_target = [Qnet(map_size**2, 4) for _ in range(num_lifts + num_robots)]
    # q = Qnet()
    # q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    reward_return_list = []
    for n_epi in range(1000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            a = []
            for i in range(num_lifts + num_robots):
                num = q.sample_action(torch.from_numpy(s).float(), epsilon)
                # 각 에이전트의 액션을 0부터 3까지의 값으로 제한
                print("num:::",num)
                num = num % 4
                a.append(num)
            # 이 정수를 리스트로 변환하여 환경에 전달
            print("num:::",a)
            s_prime, r, dones, _ = env.step(a)

            done = dones
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size()>2000:
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            reward_return_list.append(score/print_interval)
            score = 0.0

    env.close()
    now = datetime.now()
    nowtxt = now.strftime('%Y-%m-%d%H:%M:%S')
    plt.plot(reward_return_list)
    plt.xlabel('Iteration')
    plt.ylabel('Reward Origin_DQN')
    plt.savefig('Reward_Origin_DQN'+nowtxt+'.png', format='png', dpi=300)
    # Display the plot
    plt.show()
if __name__ == '__main__':
    main()
