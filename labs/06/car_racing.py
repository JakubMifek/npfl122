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
        self.model.add(tf.keras.layers.Dense(args.actions**3))
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
        self.target.add(tf.keras.layers.Dense(args.actions**3))
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
    parser.add_argument("--episodes", default=16*4, type=int, help="Training episodes.")
    parser.add_argument("--frame_skip", default=4, type=int, help="Repeat actions for given number of frames.")
    parser.add_argument("--frame_history", default=8, type=int, help="Number of past frames to stack together.")
    parser.add_argument("--render_each", default=4, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")

    parser.add_argument("--alpha", default=0.5, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.01, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.0001, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.9, type=float, help="Discounting factor.")

    parser.add_argument("--buffer", default=1024**2, type=int, help="Replay buffer size.")
    parser.add_argument("--actions", default=3, type=int, help="To how many buckets discretize actions.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size for the NN.")
    parser.add_argument("--reset_target_each", default=32, type=int, help="How often to reset target network (in steps).")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = car_racing_evaluator.environment(args.frame_skip)

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
    network.init(env, args)

    # Current step used for updating target network "once in a while"
    step = 0

    # Current episode since env in parallel execution does not count them
    episode = 0
    
    # Initialize args.threads parallel agents
    states, dones = env.parallel_init(args.threads), [False] * args.threads

    # State history
    state_history = [[state for i in range(args.frame_history)] for state in states]

    T = time.time()

    while episode < args.episodes:
        # Determine actions
        if epsilon < np.random.rand():
            # random action for each agent
            acts = [i for i in np.random.choice(range(len(actions)), size=[args.threads])]
        else:
            # predicted action for each agent
            predictions = network.predict(state_history)
            acts = [i for i in np.argmax(predictions, axis=1)]

        # Execute all actions
        returns = env.parallel_step([actions[i] for i in acts])
        # Increase step count
        step += 1

        for i in range(len(dones)):
            next_state, reward, done, _ = returns[i]

            # Append step to the buffer
            if N < args.buffer:
                N += 1
            else:
                replay_buffer.popleft()

            # Append Transition to replay buffer
            replay_buffer.append(Transition([state for state in state_history[i]], acts[i], reward, done, (next_state if not done else None)))

            # If done => new episode automatically started
            if done:
                # reset state history
                state_history[i] = [next_state for j in range(args.frame_history)]
                # we finished an episode
                episode += 1
            else:
                # remove last state and add a the new one
                state_history[i] = state_history[i][1:]
                state_history[i].append(next_state)

            # Update state
            states[i] = next_state
        
        # Train on some images

        # Choose several random transitions (batch_size)
        chosen = [replay_buffer[idx] for idx in [np.random.randint(0, len(replay_buffer)) for i in range(args.batch_size)]] 

        # Get states
        states = [trans.state for trans in chosen]

        # Get actions
        acts = [trans.action for trans in chosen]

        # Get rewards
        rewards = [trans.reward for trans in chosen]

        # Get next states
        next_states = [trans.next_state for trans in chosen]

        # Train on chosen transitions
        network.train(states, acts, rewards, next_states, args)

        # Update epsilon and alpha
        if args.epsilon_final:
            epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
        if args.alpha_final:
            alpha = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.alpha), np.log(args.alpha_final)]))

        # If step is a multiplication of args.reset_target_each, reset target network weights to the current model's
        if step % args.reset_target_each == 0:
            # Get some statistics
            print('Episode {} in {}s'.format(episode, round(time.time()-T)))

            # Trial run
            state, done = env.reset(False), False
            R = 0
            sh = [state for i in range(args.frame_history)]
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                action = np.argmax(network.predict([sh])[0])
                state, reward, done, _ = env.step(actions[action])
                sh = sh[1:]
                sh.append(state)
                R += reward

            print('Reward {}'.format(R))

            network.reset_target_weights()
            
            # reset step to avoid overflow
            step = 0

    network.set_done(True)

    # Save network for future use
    network.save()
