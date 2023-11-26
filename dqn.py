import torch
import torch.nn as nn
import torch.optim as optim

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc = nn.Linear(state_size, action_size)

    def forward(self, state):
        return self.fc(state)

class DQN:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay

        # Neural network for Q-value approximation
        self.q_network = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        # epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_values = self.q_network(torch.FloatTensor(state))
            return torch.argmax(q_values).item()

    def update_q_network(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)

        # Q-value prediction for the current state
        q_values = self.q_network(state_tensor)
        current_q_value = q_values[action]

        # Q-value target computation
        target = reward
        if not done:
            next_q_values = self.q_network(next_state_tensor)
            target += self.discount_factor * torch.max(next_q_values).item()

        # Temporal difference error
        td_error = target - current_q_value

        # Loss and optimization
        loss = td_error ** 2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(0.01, self.epsilon)
