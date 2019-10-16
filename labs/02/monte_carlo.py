#!/usr/bin/env python3
import numpy as np

import cart_pole_evaluator

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    import math
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=7000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")

    parser.add_argument("--epsilon", default=1.0, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.001, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment()

    Q=np.zeros((env.states, env.actions))
    C=np.zeros((env.states, env.actions))
    a=(args.epsilon_final/args.epsilon) ** (1/max(args.episodes-200, 1000))
    e=args.epsilon

    while env.episode < args.episodes and (env.episode < 100 or np.mean(env._episode_returns[-100:]) < 500):
        # Perform a training episode
        state, done = env.reset(), False
        A = []
        S = [state]
        R = [0]
        if args.epsilon_final < e:
            e = a**env.episode * args.epsilon
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            E = np.random.rand()
            if E < e:
                action = np.random.randint(0, env.actions)
            else:
                action = np.argmax(Q[state])
            A.append(action)

            state, reward, done, _ = env.step(action)
            S.append(state)
            R.append(reward)

        G = 0
        for t in range(len(A)-1, -1, -1):
            G = args.gamma * G + R[t+1]
            C[S[t], A[t]] += 1
            Q[S[t], A[t]] += (G - Q[S[t], A[t]]) / C[S[t], A[t]]

    # Perform last 100 evaluation episodes
    for i in range(100):
        state, done = env.reset(start_evaluate=True), False
        while not done:
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)