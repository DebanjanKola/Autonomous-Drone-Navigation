
"""
Tabular Q-learning agent for the discrete DroneGridEnv. Uses epsilon-greedy exploration.
This file provides a simple training loop and save/load utilities.
"""

import numpy as np
import random
import os
import pickle


class QAgent:
    def __init__(self, obs_n, action_n, lr=0.1, gamma=0.99, epsilon=1.0, eps_min=0.01, eps_decay=0.995):
        self.obs_n = obs_n
        self.action_n = action_n
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        # initialize Q-table to zeros
        self.Q = np.zeros((obs_n, action_n), dtype=np.float32)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_n)
        else:
            return int(np.argmax(self.Q[state]))

    def learn(self, s, a, r, s2, done):
        q_predict = self.Q[s, a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * np.max(self.Q[s2])
        self.Q[s, a] = q_predict + self.lr * (q_target - q_predict)

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def save(self, path='q_agent.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, path='q_agent.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.__dict__.update(data)


