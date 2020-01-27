#!/usr/bin/env python3
# Team members:
# f9afcdf4-21f5-11e8-9de3-00505601122b
# 90257956-3ea2-11e9-b0fd-00505601122b
# 13926bf3-c4b8-11e8-a4be-00505601122b
#####################################
# (Martin Mares)
# (Jakub Mifek)
# (Jan Pacovsky)

import collections

import numpy as np
import tensorflow as tf

import gym_evaluator


class Network:
    def __init__(self, env, args):
        assert len(env.action_shape) == 1
        self.tau = args.target_tau
        action_components = env.action_shape[0]
        action_lows, action_highs = map(np.array, env.action_ranges)
        action_diff = np.array(list(map(lambda i: action_highs[i] - action_lows[i], range(len(action_lows)))))

        mul_tensor = tf.expand_dims(tf.constant(action_diff), 0)
        add_tensor = tf.expand_dims(tf.constant(action_lows), 0)

        # Create `actor` network, starting with `inputs` and returning
        # `action_components` values for each batch example. Usually, one
        # or two hidden layers are employed. Each `action_component[i]` should
        # be mapped to range `[actions_lows[i]..action_highs[i]]`, for example
        # using `tf.nn.sigmoid` and suitable rescaling.
        input_layer = tf.keras.layers.Input(shape=env.state_shape)
        hidden = tf.keras.layers.Dense(args.hidden_layer, activation='relu')(input_layer)
        hidden = tf.keras.layers.Dense(action_components, activation='sigmoid')(hidden)
        output_layer = tf.keras.layers.Multiply()([hidden, mul_tensor])
        output_layer = tf.keras.layers.Add()([output_layer, add_tensor])

        self.actor = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])
        self.actor.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss='mse',
            experimental_run_tf_function=False
        )

        print(self.actor.summary())

        # Then, create a target actor as a copy of the model using
        # `tf.keras.models.clone_model`.
        self.target_actor = tf.keras.models.clone_model(self.actor)

        # Create `critic` network, starting with `inputs` and `actions`
        # and producing a vector of predicted returns. Usually, `inputs` are fed
        # through a hidden layer first, and then concatenated with `actions` and fed
        # through two more hidden layers, before computing the returns.
        hidden = tf.keras.layers.Dense(args.hidden_layer, activation='relu')(input_layer)
        action_layer = tf.keras.layers.Input(shape=env.action_shape)
        hidden = tf.keras.layers.Concatenate()([hidden, action_layer])
        hidden = tf.keras.layers.Dense(args.hidden_layer, activation='relu')(hidden)
        hidden = tf.keras.layers.Dense(args.hidden_layer, activation='relu')(hidden)
        output_layer = tf.keras.layers.Dense(1)(hidden)

        self.critic = tf.keras.models.Model(inputs=[input_layer, action_layer], outputs=[output_layer])
        self.critic.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss='mse'
        )

        print(self.critic.summary())

        # Then, create a target critic as a copy of the model using `tf.keras.models.clone_model`.
        self.target_critic = tf.keras.models.clone_model(self.critic)
        self._actor_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    @tf.function
    def _train(self, states, actions, returns):
        with tf.GradientTape() as tape:
            actions = self.actor(states, training=True)
            values = self.critic((states, actions), training=False)[:, 0]
            loss = -tf.math.reduce_mean(values)

        actor_grads = tape.gradient(loss, self.actor.variables)
        self._actor_optimizer.apply_gradients(zip(actor_grads, self.actor.variables))

        # TODO: Train separately the actor and critic.

        # Furthermore, update the weights of the target actor and critic networks
        # by using args.target_tau option.

    def train(self, states, actions, returns):
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.float32), np.array(returns,
                                                                                                         np.float32)
        self.critic.train_on_batch((states, actions), returns)

        self._train(states, actions, returns)

        self.target_actor.set_weights(
            (1 - self.tau) * np.array(self.target_actor.get_weights()) + self.tau * np.array(self.actor.get_weights()))
        self.target_critic.set_weights(
            (1 - self.tau) * np.array(self.target_critic.get_weights()) + self.tau * np.array(
                self.critic.get_weights()))

    @tf.function
    def _predict_actions(self, states):
        # TODO: Compute actions by the actor
        return self.actor(states, training=False)

    def predict_actions(self, states):
        states = np.array(states, np.float32)
        return self._predict_actions(states).numpy()

    @tf.function
    def _predict_values(self, states):
        # TODO: Predict actions by the target actor and evaluate them using
        # target_critic.
        actions = self.target_actor(states, training=False)
        return self.target_critic((states, actions), training=False)

    def predict_values(self, states):
        states = np.array(states, np.float32)
        return self._predict_values(states).numpy()[:, 0]


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, mu, theta, sigma):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return self.state


if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--env", default="BipedalWalker-v2", type=str, help="Environment.")
    parser.add_argument("--evaluate_each", default=10, type=int, help="Evaluate each number of episodes.")
    parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of batches.")
    parser.add_argument("--noise_sigma", default=0.2, type=float, help="UB noise sigma.")
    parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=50, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=10, type=int, help="Render some episodes.")
    parser.add_argument("--target_tau", default=1e-2, type=float, help="Target network update weight.")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
    args, _ = parser.parse_known_args()

    # Fix random seeds and number of threads
    np.random.seed(128)
    tf.random.set_seed(128)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = gym_evaluator.GymEnvironment(args.env)
    action_lows, action_highs = map(np.array, env.action_ranges)

    best_result = 99

    # Construct the network
    network = Network(env, args)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    noise = OrnsteinUhlenbeckNoise(env.action_shape[0], 0., args.noise_theta, args.noise_sigma)
    while True:
        # Training
        for _ in range(args.evaluate_each):
            state, done = env.reset(), False
            noise.reset()
            while not done:
                # TODO: Perform an action and store the transition in the replay buffer
                action = np.clip(network.predict_actions([state])[0] + noise.sample(), action_lows, action_highs)
                next_state, reward, done, _ = env.step(action)

                replay_buffer.append(Transition(state, action, reward, done, next_state))
                state = next_state

                # If the replay_buffer is large enough, perform training
                if len(replay_buffer) >= args.batch_size:
                    batch = np.random.choice(len(replay_buffer), size=args.batch_size, replace=False)
                    states, actions, rewards, dones, next_states = zip(*[replay_buffer[i] for i in batch])
                    # TODO: Perform the training
                    predicted = network.predict_values(next_states)
                    returns = rewards + args.gamma * np.array(
                        [(predicted[idx] if not dones[idx] else 0) for idx in range(args.batch_size)])
                    network.train(states, actions, returns)

                if len(replay_buffer) >= 100000:
                    replay_buffer.popleft()

        # Periodic evaluation
        returns = []
        for _ in range(args.evaluate_for):
            returns.append(0)
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and (env.episode + 1) % args.render_each == 0:
                    env.render()

                action = network.predict_actions([state])[0]
                state, reward, done, _ = env.step(action)
                returns[-1] += reward

        print("Evaluation of {} episodes: {}".format(args.evaluate_for, np.mean(returns)))
        # Save networks if perform good
        if round(np.mean(returns)) > best_result: # optimally np.mean(env._episode_returns[-10:]), 2)
            print('saving model')
            best_result = round(np.mean(returns), 2)
            network.actor.save('networks/walker-actor-{}.model'.format(best_result))
            network.critic.save('networks/walker-critic-{}.model'.format(best_result))

        if best_result > 150:
            break

    # On the end perform final evaluations with `env.reset(True)`
    for _ in range(100):
        state, done = env.reset(True), False
        while not done:
            action = network.predict_actions([state])[0]
            state, reward, done, _ = env.step(action)
