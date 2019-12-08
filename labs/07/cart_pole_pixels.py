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
import cart_pole_pixels_evaluator
import time

T = time.time()

class Network:
    def __init__(self, env, args):
        # Define suitable model. Apart from the model defined in `reinforce`,
        # define also another model `baseline`, which produces one output
        # (using a dense layer without activation).
        #
        # Use Adam optimizer with given `args.learning_rate` for both models.
        input_layer = tf.keras.layers.Input(shape=env.state_shape)
        conv = tf.keras.layers.MaxPool2D(4, 2)(input_layer)
        conv = tf.keras.layers.Conv2D(16,3,2, padding='same')(conv)
        conv = tf.keras.layers.MaxPool2D(4, 2)(conv)
        conv = tf.keras.layers.Dropout(0.7)(conv)
        conv = tf.keras.layers.Flatten()(conv)

        output_layer = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(conv)
        output_layer = tf.keras.layers.Dense(env.actions, activation='softmax')(output_layer)
        self.model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            experimental_run_tf_function=False
        )

        # baseline_output = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(conv)
        # baseline_output = tf.keras.layers.Dense(1, activation=None)(baseline_output)

        # self.baseline = tf.keras.Model(inputs=[input_layer], outputs=[baseline_output])
        
        # self.baseline.compile(
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        #     loss='mse',
        #     experimental_run_tf_function=False
        # )

        print(self.model.summary())
        # print(self.baseline.summary())

    def train(self, states, actions, returns):
        actions = np.array(list(reduce(lambda a, b: np.concatenate((a,b)), actions)))
        returns = np.array(list(reduce(lambda a, b: np.concatenate((a,b)), returns)))
        states = np.array(list(reduce(lambda a, b: np.concatenate((a,b)), states)))

        # Train the model using the states, actions and observed returns.
        # You should:
        # - compute the predicted baseline using the `baseline` model
        # baselines = self.baseline.predict(states).flatten()

        # - train the policy model, using `returns - predicted_baseline` as weights
        #   in the sparse crossentropy loss
        weight = returns # - baselines
        self.model.train_on_batch(states, actions, sample_weight=weight)

        # - train the `baseline` model to predict `returns`
        # self.baseline.train_on_batch(states, returns)

    def predict(self, states):
        states = np.array(states, np.float32)
        return self.model.predict(states)



if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=3, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=1024*2, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.002, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = cart_pole_pixels_evaluator.environment()

    # Construct the network
    network = Network(env, args)
    A = np.array(range(env.actions))
    N = args.episodes // args.batch_size

    training = True

    best_result = 200

    # Training
    for n in range(N):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            print('Episode {}/{}'.format(env.episode+1, args.episodes))
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

        print('Reward {} -- mean[-10:] {} -- in {}s'.format(env._episode_returns[-1], np.mean(env._episode_returns[-10:]), round(time.time() - T, 2)))

        # print('Last return: {}'.format(round(np.mean(env._episode_returns[-args.batch_size:]), 2)))

        if round(np.mean(env._episode_returns[-10:]), 2) > best_result:
            print('saving model')
            best_result = round(np.mean(env._episode_returns[-10:]), 2)
            network.model.save('networks/reinforce-pixels-{}.model'.format(best_result))

        if not training:
            break

        # Train using the generated batch
        network.train(
            batch_states,
            batch_actions,
            batch_returns
        )
        # print('Training {}/{} done in {}s'.format(n+1, N, round(time.time() - T, 2)))

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        # R = 0
        while not done:
            # Compute action `probabilities` using `network.predict` and current `state`

            # Choose greedy action this time
            probabilities = network.predict([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)
            # R += reward
