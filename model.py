import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Linear_QNet(nn.Module):
    def __init__(self, input_size=9, hidden_size=256, output_size=9):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()


    def train_step(self, states, actions, rewards, next_states, is_terminals):
        # Convert to tensors with proper batch dimensions
        states = torch.tensor(np.array(states).reshape(-1, 9)).float()
        next_states = torch.tensor(np.array(next_states).reshape(-1, 9)).float()
        actions = torch.tensor(actions).long().view(-1, 1)
        rewards = torch.tensor(rewards).float().view(-1, 1)
        is_terminals = torch.tensor(is_terminals).bool()

        # 1. Get current Q values for chosen actions
        current_q = self.model(states).gather(1, actions)

        # 2. Calculate target Q values
        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0].view(-1,1)
            target_q = rewards + (self.gamma * next_q * ~is_terminals)

        # 3. Compute loss
        loss = self.criterion(current_q, target_q)

        # 4. Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()