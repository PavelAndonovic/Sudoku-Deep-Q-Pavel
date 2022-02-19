import time
import pandas as pd
import numpy as np
import os


# Data Pre-processing
def split(word):
    return [char for char in word]


COLUMNS = ['quizzes', 'solutions']
PATH = "./sudoku.csv"
df_train = pd.read_csv(PATH,
                       skipinitialspace=True,
                       names=COLUMNS,
                       index_col=False)

# Data Frame type
quizzes = df_train['quizzes'].astype(str)
solutions = df_train['solutions'].astype(str)


class board(object):
    def __init__(self):
        super(board, self).__init__()
        # Choose Actions
        self.action_space = ['up', 'down', 'lelf', 'right', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.n_actions = len(self.action_space)
        # Features in Neural Network
        self.n_features = 2
        self.mazecount = 1
        self._build_maze()

    def _build_maze(self):
        # Making Environment
        self.currentquiz = quizzes.iloc[self.mazecount]
        self.quizreshaped = np.asarray(self.currentquiz)
        self.quizarray = split(str(self.quizreshaped))
        # Generating binary sudoku array for making fixed sudoku maze
        self.binaryquiz = []
        for i in self.quizarray:
            if (i == '0'):
                self.binaryquiz.append('0')
            else:
                self.binaryquiz.append('1')
        self.quizarray = np.array(self.quizarray).reshape(9, 9)
        self.binaryquizarray = np.array(self.binaryquiz).reshape(9, 9)
        self.agent = np.array([0, 0])

        # Extract Solution of current question
        self.currentsolution = solutions.iloc[self.mazecount]
        self.solutionreshaped = np.asarray(self.currentsolution)
        self.solutionarray = split(str(self.solutionreshaped))
        self.solutionarray = np.array(self.solutionarray).reshape(9, 9)
        self.mazecount += 1

    def reset(self):
        time.sleep(0.1)
        # setting the agent pointing to 0,0 position on reset
        return (np.array([0, 0]))

    def step(self, action):
        s = self.agent
        stemp = s
        if action == 0:  # up
            if (stemp[0] > 0):
                stemp[0] = stemp[0] - 1
        elif action == 1:  # down
            if (stemp[0] < 8):
                stemp[0] = stemp[0] + 1
        elif action == 2:  # right
            if (stemp[1] < 8):
                stemp[1] = stemp[1] + 1
        elif action == 3:  # left
            if (stemp[1] > 0):
                stemp[1] = stemp[1] - 1
        elif action == 4:  # insert 1
            if (self.binaryquizarray[s[0], s[1]] == '0'):
                self.quizarray[s[0], s[1]] = '1'
        elif action == 5:  # insert 2
            if (self.binaryquizarray[s[0], s[1]] == '0'):
                self.quizarray[s[0], s[1]] = '2'
        elif action == 6:  # insert 3
            if (self.binaryquizarray[s[0], s[1]] == '0'):
                self.quizarray[s[0], s[1]] = '3'
        elif action == 7:  # insert 4
            if (self.binaryquizarray[s[0], s[1]] == '0'):
                self.quizarray[s[0], s[1]] = '4'
        elif action == 8:  # insert 5
            if (self.binaryquizarray[s[0], s[1]] == '0'):
                self.quizarray[s[0], s[1]] = '5'
        elif action == 9:  # insert 6
            if (self.binaryquizarray[s[0], s[1]] == '0'):
                self.quizarray[s[0], s[1]] = '6'
        elif action == 10:  # insert 7
            if (self.binaryquizarray[s[0], s[1]] == '0'):
                self.quizarray[s[0], s[1]] = '7'
        elif action == 11:  # insert 8
            if (self.binaryquizarray[s[0], s[1]] == '0'):
                self.quizarray[s[0], s[1]] = '8'
        elif action == 12:  # insert 9
            if (self.binaryquizarray[s[0], s[1]] == '0'):
                self.quizarray[s[0], s[1]] = '9'
        # reward function

        if ((self.quizarray == self.solutionarray).all()):
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        if (action < 4):
            s_ = stemp
        else:
            s_ = s

        cls = lambda: os.system('cls')
        cls()
        print(self.quizarray)
        time.sleep(0.5)
        return s_, reward, done
