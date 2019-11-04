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
import random
import lunar_lander_evaluator
from functools import partial
import my_queue
import time


def sample_action(Q, state, epsilon, actions):
    return np.random.choice(actions) if random.random() < epsilon else np.argmax(Q[state])


def evaluate(Q, start_evaluate=False):
    e = lunar_lander_evaluator.environment(verbose=start_evaluate)
    for _ in range(100 if start_evaluate else 20):
        state, done = e.reset(start_evaluate=start_evaluate), False
        while not done:
            action = np.argmax(Q[state])
            state, _, done, _ = e.step(action)


    return np.mean(e._episode_returns[-20:]) if not start_evaluate else 0


def update_Q(Q, queue, state, gamma, alpha):
    queue.reverse_iteration()

    G = 0 if state == None else max(Q[state])
    for _, _, r in queue:
        G = gamma * G + r

    state, action, _ = queue.dequeue()
    Q[state, action] += alpha * (G - Q[state, action])

    queue.reverse_iteration()


def perform_classic_episode(env, Q, n, update_Q, get_action):
    state, done = env.reset(), False

    queue = my_queue.Queue()

    while not done:
        action = get_action(Q, state)
        next_state, reward, done, _ = env.step(action)

        queue.enqueue((state, action, reward))
        state = next_state

        if len(queue) == n:
            update_Q(Q, queue, state)

    while len(queue) > 0:
        update_Q(Q, queue, None)


def perform_expert_episode(env, Q, n, update_Q):
    state, trajectory = env.expert_trajectory()
    queue = my_queue.Queue()
    for action, reward, next_state in trajectory:
        queue.enqueue((state, action, reward))
        state = next_state

        if len(queue) == n:
            update_Q(Q, queue, state, alpha=0.2)

    while len(queue) > 0:
        update_Q(Q, queue, None, alpha=0.2)


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    T = time.time()
    print(time.strftime('%H:%M:%S'))

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=512*3.5+1,
                        type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=0,
                        type=int, help="Render some episodes.")
    parser.add_argument("--evaluate_each", default=None,
                        type=int, help="Evaluate and optionally save after some episodes.")

    parser.add_argument('--agents', default=2, type=int,
                        help='Number of agents')
    parser.add_argument('--pretrain_episodes', default=64, type=int,
                        help='Number of episode to train with guide before real training')
    parser.add_argument('--guided', default=0.3, type=float,
                        help='Chance for guided learning')
    parser.add_argument('--guided_final', default=0.00001,
                        type=float, help='Final chance for guided learning')

    parser.add_argument("--alpha", default=0.2,
                        type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.05,
                        type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.1,
                        type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.01,
                        type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.98, type=float,
                        help="Discounting factor.")
    parser.add_argument("--lookback", default=4, type=float,
                        help="Number of lookback states.")
    args = parser.parse_args()

    # Create the environment
    env = lunar_lander_evaluator.environment(verbose=True)

    # Prepare parameters
    d_eps = args.epsilon_final
    c_eps = args.epsilon
    a_eps = (d_eps-c_eps)/(args.episodes**2)
    b_eps = 0

    d_alp = args.alpha_final
    c_alp = args.alpha
    a_alp = (d_alp-c_alp)/(args.episodes**2)
    b_alp = 0

    d_gui = args.guided_final
    c_gui = args.guided
    a_gui = (d_gui-c_gui)/(args.episodes**2)
    b_gui = 0

    Qs = []
    for i in range(args.agents):
        Qs.append(np.zeros((env.states, env.actions)))

    epsilon = args.epsilon
    alpha = args.alpha
    guided = args.guided
    e_2 = 0

    get_action = partial(sample_action, actions=env.actions)
    update_policy = partial(update_Q, gamma=args.gamma)

    best_Q = np.zeros((env.states, env.actions))
    best_returns = -np.inf

    up = partial(update_policy, alpha=0.2)
    for ep in range(args.pretrain_episodes):
        if ep % 10 == 0:
            print(ep)
        
        for Q in Qs:
            perform_expert_episode(env, Q, args.lookback, up)

    while env.episode < args.episodes:
        if epsilon > args.epsilon_final:
            epsilon = max(args.epsilon_final, a_eps *
                          e_2 + b_eps*env.episode + c_eps)

        if alpha > args.alpha_final:
            alpha = max(args.alpha_final, a_alp*e_2 +
                        b_alp*env.episode + c_alp)

        if guided > args.guided_final:
            guided = max(args.guided_final, a_gui *
                         e_2 + b_gui*env.episode + c_gui)

        Q = Qs[env.episode % len(Qs)]
        if random.random() < guided:
            # perform guided episode
            perform_expert_episode(
                env, Q, args.lookback,
                partial(update_policy, alpha=alpha)
            )
        else:
            # Perform a training episode
            perform_classic_episode(
                env, Q, args.lookback,
                partial(update_policy, alpha=alpha),
                partial(get_action, epsilon=epsilon)
            )

        if args.evaluate_each and env.episode % args.evaluate_each == 0:
            Q = np.sum(Qs, axis=0)
            returns = evaluate(Q)
            print('Avg returns after {} episodes: {}'.format(env.episode, returns))
            if returns > best_returns:
                best_Q = np.copy(Q)
                best_returns = returns
        elif not args.evaluate_each and env.episode == args.episodes-1:
            best_Q = np.sum(Qs, axis=0)


        e_2 = env.episode ** 2

    # Perform last 100 evaluation episodes
    print('Computed in {:.2f}seconds'.format(time.time() - T))
    evaluate(best_Q, start_evaluate=True)
