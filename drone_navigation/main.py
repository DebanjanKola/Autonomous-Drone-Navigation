
"""
Training script to run Q-learning on DroneGridEnv.
Usage examples:
    python main.py --train --episodes 5000
    python main.py --render --episodes 1 --render_every 1

The script will save the trained Q-table to q_agent.pkl by default.
"""

import argparse
import gym
import numpy as np
import time
from q_agent import QAgent
from gym_drone.envs.drone_env import DroneGridEnv
import os

def train(env, agent, episodes=1000, render_every=0, save_path='q_agent.pkl'):
    best_steps = None
    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            a = agent.act(s)
            s2, r, done, _ = env.step(a)
            agent.learn(s, a, r, s2, done)
            s = s2
            total_reward += r
            steps += 1
            if render_every and ep % render_every == 0:
                env.render()

        agent.decay_epsilon()

        if ep % 100 == 0:
            print(f"Episode {ep} reward={total_reward:.1f} steps={steps} eps={agent.epsilon:.3f}")

    print("Training finished. Saving agent...")
    agent.save(save_path)


def evaluate(env, agent, episodes=10, render=False):
    success = 0
    steps_list = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        steps = 0
        while not done:
            a = np.argmax(agent.Q[s])
            s, r, done, _ = env.step(a)
            steps += 1
            if render:
                env.render()
        if r > 0:
            success += 1
        steps_list.append(steps)
    print(f"Eval: success {success}/{episodes}, avg_steps={np.mean(steps_list):.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--cell_size', type=int, default=16)
    parser.add_argument('--render_every', type=int, default=0)
    parser.add_argument('--qpath', type=str, default='q_agent.pkl')

    args = parser.parse_args()

    env = DroneGridEnv(cell_size=args.cell_size)
    obs_n = env.observation_space.n
    action_n = env.action_space.n
    agent = QAgent(obs_n, action_n)

    if args.train:
        train(env, agent, episodes=args.episodes, render_every=args.render_every, save_path=args.qpath)
        print('training completed')

    if args.render:
        # load saved agent if exists
        if os.path.exists(args.qpath):
            agent.load(args.qpath)
        evaluate(env, agent, episodes=5, render=True)


