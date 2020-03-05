import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable
import numpy as np


class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.n

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        action_scores = self.linear2(x)
        return F.softmax(action_scores, dim=-1)


class PolicyGradient(object):
    def __init__(self, hidden_size, num_inputs, action_space):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()
        self.cost_his = []

    def select_action(self, state):
        prob_weights = self.model(state).detach().numpy()[0]
        action = np.random.choice(range(self.action_space.n), p=prob_weights)
        return action, prob_weights

    def update_parameters(self, batch_states, batch_rewards, batch_actions):
        self.optimizer.zero_grad()
        state_tensor = torch.FloatTensor(batch_states)
        reward_tensor = torch.FloatTensor(batch_rewards).view(-1, 1)
        # Actions are used as indices, must be
        # LongTensor
        action_tensor = torch.LongTensor(batch_actions).view(-1, 1)

        # Calculate loss
        logprob = torch.log(self.model(state_tensor))
        selected_logprobs = reward_tensor * torch.gather(logprob, 1, action_tensor).squeeze()
        loss = -selected_logprobs.mean()
        self.cost_his.append(loss)
        # Calculate gradients
        loss.backward()
        # Apply gradients
        self.optimizer.step()

    def discount_rewards(self, rewards, gamma=0.99):
        r = np.array([gamma ** i * rewards[i]
                      for i in range(len(rewards))])
        # Reverse the array direction for cumsum and then
        # revert back to the original order
        r = r[::-1].cumsum()[::-1]
        return r - r.mean()

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


