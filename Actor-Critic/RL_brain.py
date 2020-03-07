import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os
import gym


class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out), dim=-1)
        return out


class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class ActorCritic(object):
    def __init__(
            self,
            n_states,
            n_actions,
            gamma = 0.99,
            learning_rate=0.01,
            batch_size=32):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size

        self._build_net()

        # cost history
        self.cost_his = []

    def _build_net(self):
        self.value_network = ValueNetwork(
            input_size=self.n_states, hidden_size=40, output_size=1)
        self.actor_network = ActorNetwork(
            input_size=self.n_states, hidden_size=40, action_size=self.n_actions)
        self.value_network_optim = torch.optim.Adam(
            self.value_network.parameters(), lr=self.lr)
        self.actor_network_optim = torch.optim.Adam(
            self.actor_network.parameters(), lr=self.lr)

    def choose_action(self, state):
        log_softmax_action = self.actor_network(Variable(torch.Tensor([state])))
        softmax_action = torch.exp(log_softmax_action)
        action = np.random.choice(self.n_actions, p=softmax_action.data.numpy()[0])
        return action

    def learn(self, actions, states, rewards):
        # final_r = self.value_network(Variable(torch.Tensor([next_state]))).data.numpy()

        actions_var = Variable(torch.Tensor(actions).view(-1, self.n_actions))
        states_var = Variable(torch.Tensor(states).view(-1, self.n_states))

        # train actor network
        self.actor_network_optim.zero_grad()
        log_softmax_actions = self.actor_network(states_var)
        vs = self.value_network(states_var).detach()
        # calculate qs
        qs = Variable(torch.Tensor(self.discount_reward(rewards))).view(-1, 1)

        advantages = qs - vs
        actor_network_loss = -torch.mean(
            torch.sum(log_softmax_actions * actions_var, 1) * advantages)
        self.cost_his.append(actor_network_loss)
        actor_network_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 0.5)
        self.actor_network_optim.step()

        # train value network
        self.value_network_optim.zero_grad()
        target_values = qs
        values = self.value_network(states_var)
        criterion = nn.MSELoss()
        value_network_loss = criterion(values, target_values)
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
        self.value_network_optim.step()

    def discount_reward(self, rewards):
        discounted_r = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

