#!/usr/bin/env python3
#Team members:
#f9afcdf4-21f5-11e8-9de3-00505601122b
#90257956-3ea2-11e9-b0fd-00505601122b
#13926bf3-c4b8-11e8-a4be-00505601122b
#####################################
#(Martin Mares)
#(Jakub Mifek)
#(Jan Pacovsky)

import numpy as np
import time
import random
import mountain_car_evaluator


def do_100_episodes(env, Q, start_evaluate):
    # Perform last 100 evaluation episodes
    for _ in range(100):
        state, done = env.reset(start_evaluate=start_evaluate), False
        while not done:
            action = np.argmax(Q[state])
            state, _, done, _ = env.step(action)


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=1024*1.5,
                        type=int, help="Training episodes.")
    parser.add_argument("--repeats", default=3,
                        type=int, help="Number of repeated tries.")
    parser.add_argument("--render_each", default=None,
                        type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.8,
                        type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.01,
                        type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.3,
                        type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.0001,
                        type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float,
                        help="Discounting factor.")
    args = parser.parse_args()

    T = time.time()

    d_eps = args.epsilon_final
    c_eps = args.epsilon
    a_eps = (d_eps-c_eps)/(args.episodes**2)
    b_eps = 0

    d_alp = args.alpha_final
    c_alp = args.alpha
    a_alp = (d_alp - c_alp) / (args.episodes**2)
    b_alp = 0

    env = mountain_car_evaluator.environment()
    Q = np.zeros((env.states, env.actions))
    
    for repeat in range(args.repeats):
        if env and env.episode > 100 and np.mean(env._episode_returns[-100:]) > -150:
            break

        # Create the environment
        env = mountain_car_evaluator.environment()
        epsilon = args.epsilon
        alpha = args.alpha
        
        while env.episode < args.episodes:
            if env.episode and env.episode % 256 == 255 and np.mean(env._episode_returns[-100:]) > -200:
                do_100_episodes(env, Q, False)
                args.episodes += 100
                if np.mean(env._episode_returns[-100:]) > -150:
                    break
                
            # Perform a training episode
            state, done = env.reset(), False
            if args.epsilon_final < epsilon:
                epsilon = a_eps*(env.episode**2) + b_eps*env.episode + c_eps

            if args.alpha_final < alpha:
                alpha = a_alp*(env.episode**2) + b_alp*env.episode + c_alp

            while not done:
                if args.render_each and env.episode and env.episode % args.render_each == 0:
                    env.render()

                if random.random() < epsilon:
                    action = np.random.choice(env.actions)
                else:
                    action = np.argmax(Q[state])

                next_state, reward, done, _ = env.step(action)

                Q[state, action] += alpha * \
                    (reward + args.gamma * max(Q[next_state]) - Q[state, action])
                state = next_state

    # Perform last 100 evaluation episodes
    print('Computed in {:.2f}seconds'.format(time.time() - T))
    do_100_episodes(env, Q, True)
