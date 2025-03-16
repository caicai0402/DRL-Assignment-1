import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class LSTMDQN(nn.Module):
    def __init__(self, state_dim=14, action_dim=6, hidden_dim=128, lr=0.01):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, action_dim)
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, x, hidden_state):
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        q_values = self.fc(lstm_out[:, -1, :])
        return q_values, hidden_state

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))

    def update(self, memory, batch_size, gamma):
        if len(memory) < batch_size:
            return
        
        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        # Compute Q values
        q_values, _ = self.forward(states, self.init_hidden(batch_size))
        q_values = q_values.gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            next_q_values, _ = self.forward(next_states, self.init_hidden(batch_size))
            max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            target_q_values = rewards + gamma * max_next_q_values * (1 - dones)
        
        # Compute loss
        loss = self.criterion(q_values, target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state, epsilon=0):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_dim)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dim
            with torch.no_grad():
                q_values, hidden = self.forward(state_tensor, self.init_hidden(1))
            action = q_values.argmax().item()
        return action
