from boardenv import board
from neural import DeepQNetwork
import pandas as pd

COLUMNS = ['quizzes', 'solutions']
PATH = "./sudoku.csv"
df_train = pd.read_csv(PATH,
                       skipinitialspace=True,
                       names=COLUMNS,
                       index_col=False)

quizzescount = 1  # df_train['quizzes'].count()


def run_maze():
    step = 0
    episodes = 10
    for episode in range(episodes):
        currentstate = env.reset()
        while True:
            action = neuralbrain.choose_action(currentstate)
            futurestate_, reward, done = env.step(action)
            neuralbrain.store_transition(currentstate, action, reward, futurestate_)
            if (step > 200) and (step % 5 == 0):
                neuralbrain.learn()
            currentstate = futurestate_
            if done:
                break
            step += 1

    print('game over')


if __name__ == "__main__":
    env = board()
    neuralbrain = DeepQNetwork(env.n_actions, env.n_features,
                               learning_rate=0.01,
                               reward_decay=0.9,
                               e_greedy=0.9,
                               replace_target_iter=200,
                               memory_size=2000)
    for i in range(1, quizzescount):
        run_maze()
        env._build_maze()
