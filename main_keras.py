from neural_keras import Agent
import numpy as np
#from utils import plotLearning
import gym
from boardenv import board

if __name__ == '__main__':
    env = board()
    n_games = 50
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=(9, 9, 1),
                  n_actions=13, mem_size=1000000, batch_size=64, epsilon_end=0.01)

    scores=[]
    eps_history=[]

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
