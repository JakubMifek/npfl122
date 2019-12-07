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
from functools import reduce
import time
import cart_pole_evaluator

T = time.time()

class Network:
    def __init__(self, env, args):
        # Create a suitable network, using Adam optimizer with given
        # `args.learning_rate`. The network should predict distribution over
        # possible actions.

        # Warning: If you plan to use Keras `.train_on_batch` and/or `.predict_on_batch`
        # methods, pass `experimental_run_tf_function=False` to compile. There is
        # a bug in TF 2.0 which causes the `*_on_batch` methods not to use `tf.function`.

        # Otherwise, if you are training manually, using `tf.function` is a good idea
        # to get good performance.
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Input(shape=env.state_shape))
        for _ in range(args.hidden_layers):
            self.model.add(tf.keras.layers.Dense(args.hidden_layer_size, activation='relu'))
            # self.model.add(tf.keras.layers.Dropout(0.1))
        self.model.add(tf.keras.layers.Dense(env.actions, activation='softmax'))
        
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=args.learning_rate),
            loss=self.loss,
            experimental_run_tf_function=False
        )

        print(self.model.summary())

    def train(self, states, actions, returns, env):        
        actions = np.array(list(reduce(lambda a, b: np.concatenate((a,b)), actions)))
        returns = np.array(list(reduce(lambda a, b: np.concatenate((a,b)), returns)))
        states = np.array(list(reduce(lambda a, b: np.concatenate((a,b)), states)))

        # Train the model using the states, actions and observed returns.
        # Use `returns` as weights in the sparse crossentropy loss.
        self.model.train_on_batch(states, actions, sample_weight=returns)

    def predict(self, states):
        states = np.array(states, np.float32)
        return self.model.predict(states)

    def save(self, value):
        self.model.save('networks/reinforce-{}-{}.model'.format(int(time.time()), value))


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=15, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=1000*2, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=0.98, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layers", default=3, type=int, help="Number of hidden layers.")
    parser.add_argument("--hidden_layer_size", default=32, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.05, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument('--test', default=30, type=int, help="How often to test the network.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)
    env2 = cart_pole_evaluator.environment(discrete=False)
    print(env.actions)
    print(env.state_shape)
    print(env.action_shape)

    # Construct the network
    network = Network(env, args)
    A = np.array(range(env.actions))
    N = args.episodes // args.batch_size

    training = True

    # Training
    for n in range(N):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset(), False

            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                # Compute action probabilities using `network.predict` and current `state`
                probabilities = network.predict([state])[0]
                S = sum(probabilities)

                # Choose `action` according to `probabilities` distribution (np.random.choice can be used)
                action = np.random.choice(A, p=probabilities/S)
                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # Compute returns by summing rewards (with discounting)

            G = 0
            Gs = []
            rewards.reverse()
            for r in rewards:
                G = r + args.gamma * G
                Gs.append(G)
            Gs.reverse()

            # Add states, actions and returns to the training batch

            batch_states.append(states)
            batch_actions.append(actions)
            batch_returns.append(Gs)

            if args.test and env.episode > 0 and np.mean(env._episode_returns[-50:]) > 470 and env.episode % args.test == 0:
                for _ in range(30):
                    state, done = env2.reset(), False
                    while not done:
                        probabilities = network.predict([state])[0]
                        action = np.argmax(probabilities)
                        state, reward, done, _ = env2.step(action)
                
                print('Test result: {}'.format(np.mean(env2._episode_returns[-60:])))

                if np.mean(env2._episode_returns[-30:]) > 495:
                    #network.save(np.mean(env2._episode_returns[-100:]))
                    print('Training finished')
                    training = False
                    break


        if not training:
            break

        # Train using the generated batch
        network.train(
            batch_states,
            batch_actions,
            batch_returns,
            env
        )
        print('Training {}/{} done in {}s'.format(n+1, N, round(time.time() - T, 2)))

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            # Compute action `probabilities` using `network.predict` and current `state`

            # Choose greedy action this time
            probabilities = network.predict([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)
        # print('Time: {}s'.format(round(time.time() - T, 2)))
