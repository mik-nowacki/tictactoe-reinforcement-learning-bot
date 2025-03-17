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
        x = self.fc3(x)  # No activation, since we want raw Q-values
        return x


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states, available_moves, actions, rewards, next_states, is_terminals):
        # Convert to tensors with proper batch dimensions
        states = torch.tensor(np.array(states).reshape(-1, 9)).float()
        next_states = torch.tensor(np.array(next_states).reshape(-1, 9)).float()
        actions = torch.tensor(actions).long().view(-1, 1)  # Shape: [batch_size, 1]
        rewards = torch.tensor(rewards).float().view(-1, 1)  # Shape: [batch_size, 1]
        is_terminals = torch.tensor(is_terminals).bool()

        # 1. Get current Q values for chosen actions
        current_q = self.model(states).gather(1, actions)  # Shape: [batch_size, 1]

        # 2. Calculate target Q values
        with torch.no_grad():
            next_q = self.model(next_states)
            
            # Mask invalid moves by setting their Q values to -inf
            for i, moves in enumerate(available_moves):
                # Convert (row,col) moves to indices
                valid_indices = [row*3 + col for (row, col) in moves]
                mask = torch.ones(9, dtype=torch.bool)
                for idx in range(9):
                    if idx not in valid_indices:
                        mask[idx] = False
                next_q[i][mask] = -float('inf')

            max_next_q = next_q.max(1)[0].view(-1, 1)  # Shape: [batch_size, 1]
            target_q = rewards + (self.gamma * max_next_q * ~is_terminals.view(-1, 1))

        # 3. Compute loss
        loss = self.criterion(current_q, target_q)

        # 4. Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()