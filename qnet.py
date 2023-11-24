# libraries
import gymnasium as gym
import collections
import random
import numpy as np
from env import WareHouseEnv
# pytorch library is used for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self, num_lift, num_robot, num_actions_per_agent):
        super(Qnet, self).__init__()
        self.num_agents = num_lift + num_robot
        self.num_actions_per_agent = num_actions_per_agent
        input_size = num_lift * 2 + num_robot * 2 + num_actions_per_agent  # Update according to your state size
        output_size = self.num_agents * num_actions_per_agent  # Each agent has 'num_actions_per_agent' possible actions
        
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        actions = []
        for i in range(self.num_agents):
            if random.random() < epsilon:
                actions.append(random.randint(0, self.num_actions_per_agent - 1))
            else:
                agent_actions = out[i*self.num_actions_per_agent:(i+1)*self.num_actions_per_agent]
                actions.append(agent_actions.argmax().item())
        return actions


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)

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
    num_lift = 2  # Example values
    num_robot = 3
    print_interval = 20
    num_actions_per_agent = 5  # Assuming 5 possible actions per agent
    env = WareHouseEnv(map_size=5, max_steps=2000, graphic=False)  # Assuming you have these parameters
    q = Qnet(num_lift, num_robot, num_actions_per_agent)
    q_target = Qnet(num_lift, num_robot, num_actions_per_agent)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    
    for n_epi in range(3000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))
        s = env.reset()
        score = 0.0  # Initialize score
        done = False

        while not done:
            flat_s = np.array(s).flatten()  # Flatten the state
            a = q.sample_action(torch.from_numpy(flat_s).float(), epsilon)
            s_prime, r, done, _ = env.step(a)  # Now 'a' is a list of actions
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
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
            score = 0.0

    env.close()

    