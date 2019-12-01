import tensorflow as tf
import car_racing_evaluator
import collections
import numpy as np

Args = collections.namedtuple("Args", ["actions", "frame_skip", "frame_history"])


args = Args(3, 4, 8)

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

env = car_racing_evaluator.environment(args.frame_skip)

model = tf.keras.models.load_model('./networks/1575216636.model')

# Final evaluation
for i in range(15):
    state, done = env.reset(True), False
    sh = [state for i in range(args.frame_history)]
    R = 0
    while not done:
        preds = model.predict([[sh]])
        action = np.argmax(preds, axis=1)[0]
        state, reward, done, _ = env.step(actions[action])
        sh = sh[1:]
        sh.append(state)
        R += reward
    print(R)
