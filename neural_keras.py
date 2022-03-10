from keras.layers import Dense, Activation
from keras.models import sequential, load_model
from keras.optimizer_v1 import Adam
import numpy as np
import gym
import gym_sudoku
env = gym.make('Sudoku-v0')