#!/usr/bin/env python3
import numpy as np
from functools import partial
import mountain_car_evaluator

def get_estimate(state, W):
    return np.mean([W[i] for i in state], axis=0)

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.2, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.001, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.0001, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    parser.add_argument("--tiles", default=2, type=int, help="Number of tiles.")
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment(tiles=args.tiles)

    # Implement Q-learning RL algorithm, using linear approximation.
    W = np.zeros([env.weights, env.actions])
    print(env.weights) # -- 631 (tiles = 8) ; 307 (tiles = 4)
    # print(env.states)  -- 631
    # print(env.actions) -- 3
    epsilon = args.epsilon
    alpha = args.alpha / args.tiles
    M = 0

    evaluating = False
    while not evaluating:
        # Perform a training episode
        state, done = env.reset(evaluating), False
        
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            # TODO: Choose `action` according to epsilon-greedy strategy
            estimate = get_estimate(state, W)
            if np.random.rand() < epsilon:
                action = np.random.choice(env.actions)
            else:
                action = np.argmax(estimate)

            next_state, reward, done, _ = env.step(action)

            # TODO: Update W values
            delta = (reward + args.gamma * max(get_estimate(next_state, W)) - estimate[action])
            for i in state:
                W[i] += alpha * delta

            state = next_state
            if done:
                break

        # TODO: Decide if we want to start evaluating
        if env.episode >= args.episodes:
            evaluating = True

        if not evaluating:
            if args.epsilon_final:
                epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
            if args.alpha_final:
                alpha = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.alpha), np.log(args.alpha_final)])) / args.tiles

    # Perform the final evaluation episodes
    while True:
        state, done = env.reset(evaluating), False
        while not done:
            # TODO: choose action as a greedy action
            action = np.argmax(get_estimate(state, W))
            state, reward, done, _ = env.step(action)
