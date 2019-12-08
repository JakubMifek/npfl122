import numpy as np
import tensorflow as tf
import cart_pole_pixels_evaluator

class Network:
    def __init__(self, env):
        input_layer = tf.keras.layers.Input(shape=env.state_shape)
        conv = tf.keras.layers.MaxPool2D(4, 2)(input_layer)
        conv = tf.keras.layers.Conv2D(16,3,2, padding='same')(conv)
        conv = tf.keras.layers.MaxPool2D(4, 2)(conv)
        conv = tf.keras.layers.Dropout(0.7)(conv)
        conv = tf.keras.layers.Flatten()(conv)

        output_layer = tf.keras.layers.Dense(128, activation='relu')(conv)
        output_layer = tf.keras.layers.Dense(env.actions, activation='softmax')(output_layer)
        self.model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            experimental_run_tf_function=False
        )

        print(self.model.summary())

    def predict(self, states):
        states = np.array(states, np.float32)
        return self.model.predict(states)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)


# Create the environment
env = cart_pole_pixels_evaluator.environment()
np.random.seed(1232147)
tf.random.set_seed(1232147)
# Construct the network
network = Network(env)
network.load_weights('./networks/reinforce-pixels-451.0.weights')

for i in range(100):
    env.reset(False)
print('let\'s go')
while True:
    state, done = env.reset(True), False
    while not done:
        probabilities = network.predict([state])[0]
        action = np.argmax(probabilities)
        state, reward, done, _ = env.step(action)