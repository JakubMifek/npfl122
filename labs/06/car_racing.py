#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import collections
import car_racing_evaluator

class Network:
    def __init__(self, env, args):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(8,8), activation='relu', strides=4))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(4,4), activation='relu', strides=2))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', strides=1))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        self.model.add(tf.keras.layers.Dense(args.actions**3))
        self.model.compile(
            optimizer='sgd',
            loss='mse',
            metrics=['mse'],
            experimental_run_tf_function=False
        )

        self.target = tf.keras.models.Sequential()
        self.target.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(8,8), activation='relu', strides=4))
        self.target.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(4,4), activation='relu', strides=2))
        self.target.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', strides=1))
        self.target.add(tf.keras.layers.Flatten())
        self.target.add(tf.keras.layers.Dense(512, activation='relu'))
        self.target.add(tf.keras.layers.Dense(args.actions**3))
        self.target.compile()
        # TODO: Create a suitable network

        # Warning: If you plan to use Keras `.train_on_batch` and/or `.predict_on_batch`
        # methods, pass `experimental_run_tf_function=False` to compile. There is
        # a bug in TF 2.0 which causes the `*_on_batch` methods not to use `tf.function`.

        # Otherwise, if you are training manually, using `tf.function` is a good idea
        # to get good performance.
        pass

    # Define a training method. Generally you have two possibilities
    # - pass new q_values of all actions for a given state; all but one are the same as before
    # - pass only one new q_value for a given state, including the index of the action to which
    #   the new q_value belongs
    def train(self, states): # TODO: + params
        # TODO
        # Train ideally on batch of images
        pass

    def predict(self, states):
        # TODO
        # Predict ideally a batch of values
        pass

    def reset_target_weights(self):
        self.target.set_weights(self.model.get_weights())

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=1024*8, type=int, help="Training episodes.")
    parser.add_argument("--frame_skip", default=3, type=int, help="Repeat actions for given number of frames.")
    parser.add_argument("--frame_history", default=5, type=int, help="Number of past frames to stack together.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")

    parser.add_argument("--alpha", default=0.5, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.01, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.0001, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.95, type=float, help="Discounting factor.")

    parser.add_argument("--buffer", default=1024**2, type=int, help="Replay buffer size.")
    parser.add_argument("--actions", default=3, type=int, help="To how many buckets discretize actions.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for the NN.")
    parser.add_argument("--reset_target_each", default=1024, type=int, help="How often to reset target network (in steps).")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = car_racing_evaluator.environment(args.frame_skip)

    # Replay Buffer
    replay_buffer = collections.deque(maxlen=args.buffer)

    # Size of the buffer (in order to not use slow count)
    N = 0

    # Transition tuple
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    # Actions that can be executed
    actions = []
    a = -1
    b = 0
    c = 0
    ad = 2.0/(args.actions-1)
    bd = 1.0/(args.actions-1)
    cd = 1.0/(args.actions-1)

    for i in range(args.actions):
        for j in range(args.actions):
            for k in range(args.actions):
                actions.append([round(a+ad*i, 1), round(b+bd*j, 1), round(c+cd*k, 1)])
    
    # Epsilon
    epsilon = args.epsilon

    # Alpha
    alpha = args.alpha

    # Network to predict Q with two models - target and model
    # Model weights can be copied to target network using the
    # dedicated function `reset_target_weights`
    network = Network(env, args)

    # Current step used for updating target network "once in a while"
    step = 0

    # Current episode since env in parallel execution does not count them
    episode = 0
    
    # Initialize args.threads parallel agents
    states, dones = env.parallel_init(args.threads), [False] * args.threads
    while episode < args.episodes:
        actions = np.zeros((args.threads, 3))
        
        for i in range(len(dones)):
            if dones[i]:
                # Perform some evaluation or something
                pass

            # Determine action
            actions[i] += np.array([0, 1, 0])
        
        # Execute all actions
        returns = env.parallel_step(actions)
        # Increase step count
        step += 1

        for i in range(len(dones)):
            next_state, reward, done, _ = returns[i]
            # Append step to the buffer
            if N < args.buffer:
                N += 1
            else:
                replay_buffer.popleft()

            # If done => new episode automatically started
            if done:
                episode += 1

            # Append Transition to replay buffer
            replay_buffer.append(Transition(states[i], actions[i], reward, done, next_state))

            # TODO: Implement some reward system

            # Update state
            states[i] = next_state
        
        # Update epsilon and alpha
        if args.epsilon_final:
            epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
        if args.alpha_final:
            alpha = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.alpha), np.log(args.alpha_final)]))

        # If step is a multiplication of args.reset_target_each, reset target network weights to the current model's
        if step % args.reset_target_each == 0:
            print(episode)
            network.reset_target_weights()
            # reset step to avoid overflow
            step = 0

    # TODO: Save the network for submission


    # while not done:
    #     if args.render_each and (env.episode + 1) % args.render_each == 0:
    #         env.render()

    #     action = [0, 1, 0]
    #     next_state, reward, done, _ = env.step(action)
