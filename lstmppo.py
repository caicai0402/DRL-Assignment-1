import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMPPO(nn.Module):
    def __init__(self, state_size=16, action_size=6, hidden_in_dim=64, hidden_out_dim=32, learning_rate=0.01):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_in_dim = hidden_in_dim
        self.hidden_out_dim = hidden_out_dim
        self.memory = []
        
        self.fc   = nn.Linear(state_size, hidden_in_dim)
        self.lstm  = nn.LSTM(hidden_in_dim, hidden_out_dim)
        self.actor_fc = nn.Linear(hidden_out_dim, action_size)
        self.critic_fc  = nn.Linear(hidden_out_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_out_dim, dtype=torch.float), torch.zeros(1, 1, self.hidden_out_dim, dtype=torch.float))

    def pi(self, x, hidden):
        x = F.relu(self.fc(x))
        x = x.view(-1, 1, self.hidden_in_dim)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.actor_fc(x)
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden
    
    def v(self, x, hidden):
        x = F.relu(self.fc(x))
        x = x.view(-1, 1, self.hidden_in_dim)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.critic_fc(x)
        return v
      
    def put_data(self, transition):
        self.memory.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.memory:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                         torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.memory = []
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]
        
    def train_net(self, gamma=0.99, lmbda=0.95, epsilon_clip=0.1, epochs=3):
        s,a,r,s_prime,done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden  = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(epochs):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()
            
            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()
