#!/usr/bin/env python3
#Team members:
#f9afcdf4-21f5-11e8-9de3-00505601122b
#90257956-3ea2-11e9-b0fd-00505601122b
#13926bf3-c4b8-11e8-a4be-00505601122b
#####################################
#(Martin Mares)
#(Jakub Mifek)
#(Jan Pacovsky)

class Queue:
    def __init__(self, iter = []):
        self.first = None
        self.last = None
        self.backwards = False
        self._in_iter = False
        self.N = 0

        for item in iter:
            self.enqueue(item)

    def __iter__(self):
        self._in_iter = True
        node = self.first if not self.backwards else self.last
        while node != None:
            yield node.value
            node = node.succ if not self.backwards else node.pred
        self._in_iter = False

    def __len__(self):
        return self.N

    def reverse_iteration(self):
        if self._in_iter:
            raise "Cannot reverse when enumerating"
        self.backwards = not self.backwards

    def enqueue(self, value):
        if self._in_iter:
            raise "Cannot enqueue when enumerating"
        if self.last == None:
            self.first = self.last = Node(value)
        else:
            node = Node(value)
            node.pred = self.last
            self.last.succ = node
            self.last = node
        self.N += 1

    def dequeue(self):
        if self._in_iter:
            raise "Cannot dequeue when enumerating"
        node = self.first
        self.first = node.succ
        if self.first:
            self.first.pred = None

        self.N -= 1
        
        return node.value

class Node:
    def __init__(self, value):
        self.value = value
        self.succ = None
        self.pred = None

import numpy as np
import random
import lunar_lander_evaluator
from functools import partial
import time
import multiprocessing as mp

def sample_action(Q, state, epsilon, actions):
    return np.random.choice(actions) if random.random() < epsilon else np.argmax(Q[state])


def evaluate(Q, start_evaluate=False, output_file='output'):
    e = lunar_lander_evaluator.environment()
    for _ in range(100):
        state, done = e.reset(start_evaluate=start_evaluate), False
        while not done:
            action = np.argmax(Q[state])
            state, _, done, _ = e.step(action)

    if not start_evaluate:
        np.save(output_file+'-'+np.mean(e._episode_returns[-100:]), Q)

    return np.mean(e._episode_returns[-100:]) if not start_evaluate else 0


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

    queue = Queue()

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

    queue=Queue()
    for action, reward, next_state in trajectory:
        queue.enqueue((state, action, reward))
        state = next_state

    while len(queue) > 0:
        update_Q(Q, queue, None)


def train_Q(Q, index, args, a_eps, b_eps, c_eps, a_alp, b_alp, c_alp, a_gui, b_gui, c_gui, update_policy, get_action):
    e_2 = 0
    epsilon = args.epsilon
    alpha = args.alpha
    guided = args.guided
    env = lunar_lander_evaluator.environment()
    episodes = args.episodes + args.pretrain_episodes
    episode = 0

    for ep in range(args.pretrain_episodes):
        if ep % 10 == 0:
            print(ep)

        if epsilon > args.epsilon_final:
            epsilon = max(args.epsilon_final, a_eps *
                          e_2 + b_eps*env.episode + c_eps)

        if alpha > args.alpha_final:
            alpha = max(args.alpha_final, a_alp*e_2 +
                        b_alp*env.episode + c_alp)

        if guided > args.guided_final:
            guided = max(args.guided_final, a_gui *
                         e_2 + b_gui*env.episode + c_gui)

        perform_expert_episode(env, Q, args.lookback, 
                partial(update_policy, alpha=alpha))

        episode += 1

    while episode < episodes:
        if epsilon > args.epsilon_final:
            epsilon = max(args.epsilon_final, a_eps *
                          e_2 + b_eps*env.episode + c_eps)

        if alpha > args.alpha_final:
            alpha = max(args.alpha_final, a_alp*e_2 +
                        b_alp*env.episode + c_alp)

        if guided > args.guided_final:
            guided = max(args.guided_final, a_gui *
                         e_2 + b_gui*env.episode + c_gui)

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

        e_2 = env.episode ** 2
        episode += 1
        if args.evaluate_each and episode % args.evaluate_each == 0:
            val = evaluate(Q, False, args.output+'_{}'.format(index))
            print('Q_{} performs with score {}'.format(index, val))

    return Q

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    T = time.time()
    print(time.strftime('%H:%M:%S'))

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=1024*80,
                        type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=0,
                        type=int, help="Render some episodes.")
    parser.add_argument("--evaluate_each", default=None,
                        type=int, help="Evaluate and optionally save after some episodes.")
    parser.add_argument("--output", default="models/q_01(lb=3;ep=80e3)", type=str,
                        help="Output filename.")

    parser.add_argument('--agents', default=8,
                        type=int, help='Number of agents')
    parser.add_argument('--pretrain_episodes', default=512*10, type=int,
                        help='Number of episode to train with guide before real training')
    parser.add_argument('--guided', default=0.3, type=float,
                        help='Chance for guided learning')
    parser.add_argument('--guided_final', default=0.00001,
                        type=float, help='Final chance for guided learning')

    parser.add_argument("--alpha", default=0.2,
                        type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.0005,
                        type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.1,
                        type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.0001,
                        type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.98, type=float,
                        help="Discounting factor.")
    parser.add_argument("--lookback", default=3, type=float,
                        help="Number of lookback states.")
    args = parser.parse_args()

    # Create the environment
    env = lunar_lander_evaluator.environment()

    # Prepare parameters
    episodes = args.episodes + args.pretrain_episodes

    d_eps = args.epsilon_final
    c_eps = args.epsilon
    a_eps = (d_eps-c_eps)/(episodes**2)
    b_eps = 0

    d_alp = args.alpha_final
    c_alp = args.alpha
    a_alp = (d_alp-c_alp)/(episodes**2)
    b_alp = 0

    d_gui = args.guided_final
    c_gui = args.guided
    a_gui = (d_gui-c_gui)/(episodes**2)
    b_gui = 0

    print('Creating {} agents'.format(args.agents))
    Qs = []
    for i in range(args.agents):
        Qs.append(np.zeros((env.states, env.actions)))

    get_action = partial(sample_action, actions=env.actions)
    update_policy = partial(update_Q, gamma=args.gamma)
    train = partial(
        train_Q,
        args=args, update_policy=update_policy, get_action=get_action,
        a_eps=a_eps, b_eps=b_eps, c_eps=c_eps, a_alp=a_alp, b_alp=b_alp,
        c_alp=c_alp, a_gui=a_gui, b_gui=b_gui, c_gui=c_gui,
    )

    # Process in parallel
    pool = mp.Pool()
    Qs = pool.starmap(train, [(Qs[i], i) for i in range(len(Qs))])
    pool.close()
    pool.join()

    # Perform last 100 evaluation episodes
    print('Computed in {:.2f}seconds'.format(time.time() - T))
    np.save(args.output, np.sum(Qs, axis=0))
    evaluate(np.sum(Qs, axis=0), start_evaluate=True)
