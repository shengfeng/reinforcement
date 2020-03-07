import gym
from RL_brain import ActorCritic
import torch
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


RL = ActorCritic(
    n_states=env.observation_space.shape[0],
    n_actions=env.action_space.n)

total_steps = 0
# Set up lists to hold results
total_rewards = []
batch_rewards = []
batch_actions = []
batch_states = []
batch_counter = 1
batch_size = 10

for i_episode in range(2000):

    states = []
    rewards = []
    actions = []
    final_r = 0
    done = False
    s_0 = env.reset()

    while not done:
        env.render()

        action = RL.choose_action(s_0)
        one_hot_action = [int(k == action) for k in range(env.action_space.n)]

        s_1, r, done, info = env.step(action)
        states.append(s_0)
        rewards.append(r)
        actions.append(one_hot_action)
        s_0 = s_1

        if done:
            batch_rewards.extend(rewards)
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_counter += 1
            total_rewards.append(sum(rewards))

            if batch_counter == batch_size:
                RL.learn(batch_actions, batch_states, batch_rewards)
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
