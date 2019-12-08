#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import collections
from multiprocessing import Process, Queue
import car_racing_evaluator
import time

class Network:
    def __init__(self):
        self.done = False

    def init(self, env, args):
        # Init model
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Input(shape=[args.frame_history]+env.state_shape, batch_size=args.batch_size))
        self.model.add(tf.keras.layers.Conv3D(filters=32, kernel_size=(min(8,args.frame_history),8,8), activation='relu', strides=4))
        self.model.add(tf.keras.layers.Conv3D(filters=64, kernel_size=(1,4,4), activation='relu', strides=2))
        self.model.add(tf.keras.layers.Conv3D(filters=64, kernel_size=(1,3,3), activation='relu', strides=1))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        self.model.add(tf.keras.layers.Dense(args.actions**2 * 2))
        self.model.compile(
            optimizer='rmsprop',
            loss='mse',
            metrics=['mse'],
            experimental_run_tf_function=False
        )

        # Init target
        self.target = tf.keras.models.Sequential()
        self.target.add(tf.keras.layers.Input(shape=[args.frame_history]+env.state_shape, batch_size=args.batch_size))
        self.target.add(tf.keras.layers.Conv3D(filters=32, kernel_size=(min(8,args.frame_history),8,8), activation='relu', strides=4))
        self.target.add(tf.keras.layers.Conv3D(filters=64, kernel_size=(1,4,4), activation='relu', strides=2))
        self.target.add(tf.keras.layers.Conv3D(filters=64, kernel_size=(1,3,3), activation='relu', strides=1))
        self.target.add(tf.keras.layers.Flatten())
        self.target.add(tf.keras.layers.Dense(512, activation='relu'))
        self.target.add(tf.keras.layers.Dense(args.actions**2 * 2))
        self.target.compile(
            optimizer='rmsprop',
            loss='mse',
            metrics=['mse'],
            experimental_run_tf_function=False
        )

        # Reset weights
        self.target.set_weights(self.model.get_weights())

    def is_done(self):
        return self.done

    def set_done(self, done):
        self.done = done

    def train(self, states, actions, rewards, next_states, args):
        # Here we gather images that we predict upon
        to_predict = []
        
        for i in range(len(states)):
            if next_states[i] is None:
                to_predict.append(states[i])
            else:
                # Future state sequence
                to_predict.append(states[i][1:])
                to_predict[-1].append(next_states[i])
                
        to_predict = np.array(to_predict)
        
        # Predict current states
        Y = self.model.predict([states])

        # Predict future states
        x = self.target.predict([to_predict])
        
        # Compute target Qs
        y = [rewards[i] + args.gamma * max(x[i] if not (to_predict[i] is states[i]) else (0,0)) for i in range(args.batch_size)]
        for i in range(len(x)):
            Y[i][actions[i]] = y[i]

        # Train on batch
        self.model.train_on_batch([states], Y)


    def predict(self, states):
        # Predict given states
        return self.model.predict([states])

    def reset_target_weights(self):
        # Rewrite target weights with actual ones
        self.target.set_weights(self.model.get_weights())

    def save(self):
        # Save network with current timestamp
        self.model.save('./networks/{}.model'.format(round(time.time())))

network = Network()
replay_buffer = collections.deque()

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=215*7, type=int, help="Training episodes.")
    parser.add_argument("--frame_skip", default=4, type=int, help="Repeat actions for given number of frames.")
    parser.add_argument("--frame_history", default=6, type=int, help="Number of past frames to stack together.")
    parser.add_argument("--render_each", default=4, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=7, type=int, help="Maximum number of threads to use.")

    parser.add_argument("--alpha", default=0.5, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.01, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.001, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.9, type=float, help="Discounting factor.")

    parser.add_argument("--buffer", default=1024*16, type=int, help="Replay buffer size.")
    parser.add_argument("--actions", default=3, type=int, help="To how many buckets discretize actions.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size for the NN.")
    parser.add_argument("--reset_target_each", default=10, type=int, help="How often to reset target network (in steps).")
    parser.add_argument("--test_each", default=32, type=int, help="How often to reset target network (in steps).")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = car_racing_evaluator.environment(args.frame_skip)

    # TODO: Implement a variation to Deep Q Network algorithm.
    #
    # Example: How to perform an episode with "always gas" agent.
    state, done = env.reset(), False
    while not done:
        if args.render_each and (env.episode + 1) % args.render_each == 0:
            env.render()

        action = [0, 1, 0]
        next_state, reward, done, _ = env.step(action)

    # After training (or loading the model), you should run the evaluation:
    while True:
        state, done = env.reset(True), False
        while not done:
            # Choose greedy action
            state, reward, done, _ = env.step(action)
