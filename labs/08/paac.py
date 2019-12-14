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
import tensorflow as tf

import gym_evaluator

class Network:
    def __init__(self, env, args):
        # Similarly to reinforce, define two models:
        # - _policy, which predicts distribution over the actions
        # - _value, which predicts the value function
        # Use independent networks for both of them, each with
        # `args.hidden_layer` neurons in one hidden layer,
        # and train them using Adam with given `args.learning_rate`.
        self._policy = tf.keras.models.Sequential()
        self._policy.add(tf.keras.layers.Input(shape=env.state_shape))
        self._policy.add(tf.keras.layers.Dense(args.hidden_layer, activation='relu'))
        self._policy.add(tf.keras.layers.Dense(env.actions, activation='softmax'))

        self._policy.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            experimental_run_tf_function=False
        )

        print(self._policy.summary())

        self._value = tf.keras.models.Sequential()
        self._value.add(tf.keras.layers.Input(shape=env.state_shape))
        self._value.add(tf.keras.layers.Dense(args.hidden_layer, activation='relu'))
        self._value.add(tf.keras.layers.Dense(1))

        self._value.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss='mse',
            experimental_run_tf_function=False
        )

        print(self._value.summary())

    def train(self, states, actions, returns):
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.int32), np.array(returns, np.float32)
        # Train the policy network using policy gradient theorem
        # and the value network using MSE.
        
        values = self._value.predict_on_batch(states)[:, 0]
        deltas = returns - values
        # print(np.mean(deltas), np.mean(returns))
        # print(returns, deltas)
        # print(min(values), np.mean(values), max(values))
        # print(min(returns), np.mean(returns), max(returns))
        # print(max(abs(returns - values)))
        # print('')

        self._value.train_on_batch(states, returns)
        self._policy.train_on_batch(states, actions, sample_weight=deltas)

    def predict_actions(self, states):
        states = np.array(states, np.float32)
        return self._policy.predict_on_batch(states)

    def predict_values(self, states):
        states = np.array(states, np.float32)
        return self._value.predict_on_batch(states)[:, 0]

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment.")
    parser.add_argument("--evaluate_each", default=128, type=int, help="Evaluate each number of batches.")
    parser.add_argument("--evaluate_for", default=50, type=int, help="Evaluate for number of batches.")
    parser.add_argument("--gamma", default=0.95, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=40, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=4e-4, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=6, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--workers", default=200, type=int, help="Number of parallel workers.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(128)
    tf.random.set_seed(128)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = gym_evaluator.GymEnvironment(args.env)

    # Construct the network
    network = Network(env, args)
    A = list(range(env.actions))

    # Initialize parallel workers by env.parallel_init
    states = env.parallel_init(args.workers)
    training = True
    while training:
        # Training
        for _ in range(args.evaluate_each):
            # Choose actions using network.predict_actions
            predictions = network.predict_actions(states)
            # print(predictions)
            actions = np.array(list(map(lambda action_dist: np.random.choice(A, p=action_dist), predictions)))
            # print(actions)

            # Perform steps by env.parallel_step
            results = env.parallel_step(actions)
            next_states = [next_state for next_state, reward, done, _ in results]
            dones = [done for next_state, reward, done, _ in results]
            rewards = np.array([reward for next_state, reward, done, _ in results])
            # print(min(rewards), np.mean(rewards), max(rewards))
            # print('dones total:', np.sum([1 for _, _, done, _ in results if done]))

            # Compute return estimates by
            # - extracting next_states from steps
            # - computing value function approximation in next_states
            # - estimating returns by reward + (0 if done else args.gamma * next_state_value)
            predicted_values = network.predict_values(next_states)
            # pred2_values = network.predict_values(states)
            # print(min(predicted_values), np.mean(predicted_values), max(predicted_values))
            # print(min(pred2_values), np.mean(pred2_values), max(pred2_values))
            additions = np.array([-20 if dones[i] else predicted_values[i] for i in range(len(dones))])
            returns = rewards + (args.gamma * additions)
            # print(min(returns), np.mean(returns), max(returns))
            # print('------')

            # Train network using current states, chosen actions and estimated returns
            network.train(states, actions, returns)
            states = next_states

        # Periodic evaluation
        returns = []
        for _ in range(args.evaluate_for):
            returns.append(0)
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                probabilities = network.predict_actions([state])[0]
                action = np.argmax(probabilities)
                # print(probabilities, action)
                state, reward, done, _ = env.step(action)
                returns[-1] += reward
        print("Evaluation of {} episodes: {}".format(args.evaluate_for, np.mean(returns)))
        if np.mean(returns) > 460:
            training = False

    # On the end perform final evaluations with `env.reset(True)`
    for _ in range(100):
        state, done = env.reset(True), False
        while not done:
            probabilities = network.predict_actions([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)
