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

import cart_pole_evaluator

def do_100_episodes(env, Q, start_evaluate):
    # Perform last 100 evaluation episodes
    for i in range(100):
        state, done = env.reset(start_evaluate=start_evaluate), False
        while not done:
            action = np.argmax(Q[state])
            state, _, done, _ = env.step(action)

def mix_Q(Qs):
    R = np.zeros((Qs[0][2].states,Qs[0][2].actions))
    W = 0
    for Q, _, env in Qs:
        M = np.mean(env._episode_returns[-100:])
        w = M/500
        R += w*Q
        W += w
    return R / W

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(12564324)

    # Parse arguments
    import argparse
    import math
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=5000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")

    parser.add_argument("--epsilon", default=0.15, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.001, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment()
    env2 = cart_pole_evaluator.environment()
    env3 = cart_pole_evaluator.environment()

    Q=np.zeros((env.states, env.actions))
    C=np.zeros((env.states, env.actions))
    Q2=np.zeros((env.states, env.actions))
    C2=np.zeros((env.states, env.actions))
    Q3=np.zeros((env.states, env.actions))
    C3=np.zeros((env.states, env.actions))

    Qs = ((Q, C, env), (Q2, C2, env2), (Q3, C3, env3))
    #a = (args.epsilon_final/args.epsilon) ** (1/max(args.episodes*0.9, 1000))
    #a = (args.epsilon_final - args.epsilon)/(args.episodes * 0.9)
    d = args.epsilon_final
    c = args.epsilon
    a = (d-c)/(args.episodes**2)
    b = 0
    e = c

    T = time.time()


    while env2.episode < args.episodes:
        if env2.episode > 0 and env2.episode % 1000 == 0:
            do_100_episodes(env, mix_Q(Qs), False)
            if np.mean(env._episode_returns[-100:]) > 490:
                break

        # Perform a training episode
        if args.epsilon_final < e:
            e = a*(env.episode**2) + b*env.episode + c
        
        for Q, C, env in Qs:
            state, done = env.reset(), False
            A = []
            S = [state]
            R = [0]
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

    print('Computed in {:.2f}seconds'.format(time.time() - T))
    do_100_episodes(env, mix_Q(Qs), True)