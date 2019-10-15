#!/usr/bin/env python3
import argparse
import sys

import numpy as np

class MultiArmedBandits():
    def __init__(self, bandits, episode_length, seed=42):
        self._generator = np.random.RandomState(seed)

        self._bandits = []
        for _ in range(bandits):
            self._bandits.append(self._generator.normal(0., 1.))
        self._done = True
        self._episode_length = episode_length

    def reset(self):
        self._done = False
        self._trials = 0
        return None

    def step(self, action):
        if self._done:
            raise ValueError('Cannot step in MultiArmedBandits when there is no running episode')
        self._trials += 1
        self._done = self._trials == self._episode_length
        reward = self._generator.normal(self._bandits[action], 1.)
        return None, reward, self._done, {}

parser = argparse.ArgumentParser()
parser.add_argument('--bandits', default=10, type=int, help='Number of bandits.')
parser.add_argument('--episodes', default=100, type=int, help='Training episodes.')
parser.add_argument('--episode_length', default=1000, type=int, help='Number of trials per episode.')
parser.add_argument('--seed', default=42, type=int, help='Random seed.')

parser.add_argument('--mode', default='gradient', type=str, help='Mode to use -- greedy, ucb and gradient.')
parser.add_argument('--alpha', default=0, type=float, help='Learning rate to use (if applicable).')
parser.add_argument('--c', default=1, type=float, help='Confidence level in ucb (if applicable).')
parser.add_argument('--epsilon', default=0.1, type=float, help='Exploration factor (if applicable).')
parser.add_argument('--initial', default=0, type=float, help='Initial value function levels (if applicable).')

def greedyParams(args):
    return np.ones((args.bandits, 2)) * args.initial

def ucbParams(args):
    return np.ones((args.bandits, 2)) * args.initial

def gradientParams(args):
    return np.zeros((args.bandits,))

def distribution(params):
    E = np.exp(params)
    D = sum(E)
    return E / D

mode = {
    'greedy': greedyParams,
    'ucb': ucbParams,
    'gradient': gradientParams,
}

def main(args):
    # Fix random seed
    np.random.seed(args.seed)

    # Create environment
    env = MultiArmedBandits(args.bandits, args.episode_length)

    rewards = np.zeros(args.episodes)
    for episode in range(args.episodes):
        env.reset()

        params = mode[args.mode](args)

        steps = 0
        done = False
        while not done:
            if args.mode == 'greedy':
                e = np.random.rand()
                if e < args.epsilon:
                    action = np.random.randint(0, args.bandits)
                else:
                    actions = np.fromiter(map(lambda val: val[0], params), dtype='float')
                    action = np.where(actions == max(actions))[0][0]
            elif args.mode == 'ucb':
                actions = np.fromiter(map(lambda val: np.inf if val[1] == 0 else val[0] + args.c * np.sqrt(np.log(steps) / val[1]), params), dtype='float')
                action = np.where(actions == max(actions))[0][0]
            elif args.mode == 'gradient':
                d = distribution(params)
                p = np.random.multinomial(1, d)
                action = np.where(p == 1)[0][0]

            _, reward, done, _ = env.step(action)

            rewards[episode] += reward
            steps += 1

            if args.mode == 'greedy' or args.mode == 'ucb':
                params[action][1] += 1
                if args.alpha == 0:
                    params[action][0] += (reward - params[action][0])/params[action][1]
                else:
                    params[action][0] += args.alpha * (reward - params[action][0])
            elif args.mode == 'gradient':
                params += args.alpha * reward * (p - d)

        rewards[episode] /= steps

    return np.mean(rewards), np.std(rewards)

if __name__ == '__main__':
    mean, std = main(parser.parse_args())
    # Print the mean and std for ReCodEx to validate
    print('{:.2f} {:.2f}'.format(mean, std))
