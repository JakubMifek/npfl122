#!/usr/bin/env python3
#Team members:
#f9afcdf4-21f5-11e8-9de3-00505601122b
#90257956-3ea2-11e9-b0fd-00505601122b
#13926bf3-c4b8-11e8-a4be-00505601122b
#####################################
#(Martin Mares)
#(Jakub Mifek)
#(Jan Pacovsky)

import collections
import random
import sys

import numpy as np
import tensorflow as tf

import cart_pole_evaluator

class Network:
    def __init__(self, env, args):
        # Warning: If you plan to use Keras `.train_on_batch` and/or `.predict_on_batch`
        # methods, pass `experimental_run_tf_function=False` to compile. There is
        # a bug in TF 2.0 which causes the `*_on_batch` methods not to use `tf.function`.

        # Otherwise, if you are training manually, using `tf.function` is a good idea
        # to get good performance.
        self.train_model = tf.keras.models.Sequential()

        self.train_model.add(tf.keras.Input(shape=env.state_shape, batch_size=args.batch_size))
        for _ in range(args.hidden_layers):
            self.train_model.add(tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu))
        self.train_model.add(tf.keras.layers.Dense(env.actions, activation=tf.nn.relu))

        self.train_model.compile(
            optimizer=tf.optimizers.Adam(
                learning_rate=args.learning_rate,
                ),
            loss=tf.losses.MeanSquaredError(),
            experimental_run_tf_function=False
        )

        self.prediction_model = tf.keras.models.clone_model(self.train_model)

    # Define a training method. Generally you have two possibilities
    # - pass new q_values of all actions for a given state; all but one are the same as before
    # - pass only one new q_value for a given state, including the index of the action to which
    #   the new q_value belongs
    def train(self, states, q_values):
        self.train_model.train_on_batch(states, q_values)

    def predict(self, states):
        return self.prediction_model.predict(states)

    def update_predictions(self) -> None:
        """
        Replace prediction netwok with trained one 
        """
        self.prediction_model.set_weights(self.train_model.get_weights())

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--buffer_size", default=1_000_000, type=int, help="Replay buffer size.")
    parser.add_argument("--episodes", default=425, type=int, help="Episodes for epsilon decay.")
    parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layers", default=1, type=int, help="Number of hidden layers.")
    parser.add_argument("--hidden_layer_size", default=20, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(env, args)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque(maxlen=args.buffer_size)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon
    step = 0
    training = 0
    rewards_buffer = collections.deque(maxlen=20)

    while (np.mean(rewards_buffer) if len(rewards_buffer) > 0 else 0) < 450:
        episode_reward = 0
        # Perform episode
        state, done = env.reset(), False
        while not done:
            step += 1

            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # Compute action using epsilon-greedy policy. You can compute
            # the q_values of a given state using
            q_values = network.predict(np.array([state], np.float32))[0]

            if random.random() < epsilon:
                action = np.random.choice(range(env.actions))
            else:
                action = np.argmax(q_values)

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            # Append state, action, reward, done and next_state to replay_buffer
            replay_buffer.append(Transition(state, action, reward, done, next_state))

            # If the replay_buffer is large enough, preform a training batch
            # of `args.batch_size` uniformly randomly chosen transitions.
            if(step > args.batch_size):
                training += 1

                # Sample buffer
                mini_replay = [replay_buffer[idx] for idx in np.random.randint(len(replay_buffer), size=args.batch_size)] 

                # After you choose `states` and suitable targets, you can train the network as
                #   network.train(states, ...)
                states = np.array([trasition.state for trasition in mini_replay])
                actions = np.array([trasition.action for trasition in mini_replay])
                rewards = np.array([trasition.reward for trasition in mini_replay])
                dones = np.array([trasition.done for trasition in mini_replay])
                next_states = np.array([trasition.next_state for trasition in mini_replay])

                q_values = network.predict(states)
                next_q_values = network.predict(next_states)

                for i in range(args.batch_size):
                    if dones[i]:
                        q_values[i,actions[i]] = rewards[i]
                    else:
                        q_values[i,actions[i]] = rewards[i] + args.gamma * np.max(next_q_values[i])

                network.train(states, q_values)

                if(training % 10 == 0):
                    network.update_predictions()
                    # print(f"Prediction network updated (training={training}, epsilon={epsilon}), buffer={len(replay_buffer)}")

            state = next_state

        if args.epsilon_final:
            epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))

        rewards_buffer.append(episode_reward)
        # print(f"mean: {np.mean(rewards_buffer) }, reward: {episode_reward}")

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            action = np.argmax(network.predict(np.array([state], np.float32))[0])
            state, reward, done, _ = env.step(action)
