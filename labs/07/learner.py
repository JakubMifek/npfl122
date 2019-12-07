minbatch = 2
maxbatch = 20
batchdiff = 2
minepisodes = 180
maxepisodes = 200
episodesdiff = 10
mingamma = 0.9
maxgamma = 1.0
gammadiff = 0.025
minlayers = 1
maxlayers = 5
layersdiff = 1
minsize = 16
maxsize = 256
sizediff = 24
minlearn = 0.001
maxlearn = 0.1
learndiff = 0.001

import reinforce_baseline
import argparse
import copy
import json
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Number of episodes to train on.")
parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layers", default=3, type=int, help="Number of hidden layers.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.004, type=float, help="Learning rate.")
parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
args = parser.parse_args()

B = list(np.arange(minbatch, maxbatch, batchdiff))
E = list(np.arange(minepisodes, maxepisodes, episodesdiff))
G = list(np.arange(mingamma, maxgamma, gammadiff))
HL = list(np.arange(minlayers, maxlayers, layersdiff))
HLS = list(np.arange(minsize, maxsize, sizediff))
L = list(np.arange(minlearn, maxlearn, learndiff))

count = len(B) * len(E) * len(G) * len(HL) * len(HLS) * len(L)
print('Hyperparameter space size: {}'.format(count))

seeds = [42, 128, 256, 1234187]

def function(a, C):
    print('Testing {} with code {}'.format(a, C))

    fails = 0
    for seed in seeds:
        result = reinforce_baseline.main(a, seed)
        if result < 490:
            print('Settings {}\nF A I L E D -- {}'.format(C, result))
            fails += 1
        if result < 450 or fails > 1:
            fails = 4
            break
    
    if fails != 4:
        print('')
        print('Settings {}\n!!! S U C C E E D E D !!!'.format(C))
        print('')
    
    print('Finished {}'.format(C))

from multiprocessing import Pool, Queue

code = 0
with Pool() as pool:
    parameters = []
    for batch in B:
        for episodes in E:
            for gamma in G:
                for layers in HL:
                    for size in HLS:
                        for learn in L:
                            a = copy.deepcopy(args)
                            a.batch_size = batch
                            a.episodes = episodes
                            a.gamma = gamma
                            a.hidden_layers = layers
                            a.hidden_layer_size = size
                            a.learning_rate = learn
                            code += 1
                            parameters.append((a, code))
    pool.starmap(function, parameters)
    pool.join()