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
import cart_pole_pixels_evaluator
import time

T = time.time()

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
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
    name = './networks/reinforce-pixels-461.4.model'
    network = tf.keras.models.load_model(name)
    print('{} constructed'.format(name))
    
    # Final evaluation
    episode = 1
    while True:
        state, done = env.reset(True), False
        print('Episode {} started in {}s'.format(episode, round(time.time() - T)))
        episode += 1
        R = 0
        t1 = 0
        t1s = 0
        t2 = 0
        t2s = 0
        i = 0
        while not done:
            # env.render()
            # Compute action `probabilities` using `network.predict` and current `state`
            # Choose greedy action this time
            t1 = time.time()
            probabilities = network.predict([[state]])[0]
            t1s += time.time() - t1
            action = np.argmax(probabilities)
            t2 = time.time()
            state, reward, done, _ = env.step(action)
            t2s += time.time() - t2
            R += reward
            i += 1
            if i % 10 == 0:
                print('time on network: {}\ttime on env: {}'.format(t1s/1000, t2s/1000))
        print(R)
