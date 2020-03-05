import gym
from RL_brain import PolicyGradient
import torch
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


RL = PolicyGradient(
    hidden_size=10,
    num_inputs=env.observation_space.shape[0],
    action_space=env.action_space)

total_steps = 0
# Set up lists to hold results
total_rewards = []
batch_rewards = []
batch_actions = []
batch_states = []
batch_counter = 1
batch_size = 10

for i_episode in range(2000):

    s_0 = env.reset()
    states = []
    rewards = []
    actions = []
    done = False

    while not done:
        env.render()

        action, probs = RL.select_action(torch.Tensor([s_0]))

        s_1, r, done, info = env.step(action)
        states.append(s_0)
        rewards.append(r)
        actions.append(action)
        s_0 = s_1

        if done:
            batch_rewards.extend(RL.discount_rewards(rewards))
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_counter += 1
            total_rewards.append(sum(rewards))

            if batch_counter == batch_size:
                RL.update_parameters(batch_states, batch_rewards, batch_actions)
                batch_rewards = []
                batch_actions = []
                batch_states = []
                batch_counter = 1

                avg_rewards = np.mean(total_rewards[-100:])
                # Print running average
                print("Ep: ", i_episode + 1, "Average of last 100: %.4f" % avg_rewards)

RL.plot_cost()

plt.plot(np.arange(len(total_rewards)), total_rewards)
plt.ylabel('Cost')
plt.xlabel('training steps')
plt.show()

env.close()
