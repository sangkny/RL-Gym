'''
1. generate data or collect the data to build the neural network
2. training the given data and build model
3. confirm the model
'''

import gym
import numpy as np
import random

env = gym.make('CartPole-v1')

# N: # of cartpole plays, K: # of samples from it (K<=N), f: functions to syncronize the Cartpole,  render: display flag
def data_preparation(N, K, f, render=False):
    game_data = []
    for i in range(N):                          # N plays any way
        score = 0
        game_steps = []
        obs = env.reset()                       # reset the environment
        for step in range(K):
            if render: env.render()

            action = f(obs)                     # return the observation current status
            game_steps.append((obs, action))
            obs, reward, done, info = env.step(action)
            score += reward
            if done:
                print('Episode finished after {} time steps at episode : N {}'.format(step+1, i))
                break
        game_data.append((score, game_steps))

    game_data.sort(key=lambda s: -s[0])         # sort the game data according to score (s[0]) in a reverse order

    training_set = []
    for i in range(K):
        for step in game_data[i][1]:
            if step[1] == 0:
                training_set.append((step[0], [1, 0]))
            else:
                training_set.append((step[0], [0, 1]))

    print("{0}/{1}th score: {2}".format(K, N, game_data[K - 1][0]))
    if render:
        for i in game_data:
            print("Score: {0}".format(i[0]))
        print(training_set)
    return training_set

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
def build_model():
  model = Sequential()
  model.add(Dense(128, input_dim=4, activation='relu'))
  model.add(Dense(52, activation='relu'))
  model.add(Dense(2, activation='softmax'))
  model.compile(loss='mse', optimizer=Adam())
  return model

def train_model(model, training_set):
    X = np.array([i[0] for i in training_set]).reshape(-1,4)
    y = np.array([i[1] for i in training_set]).reshape(-1,2)
    model.fit(X, y, epochs=10)


if __name__=='__main__':
    N = 1000
    K = 50

    self_play_flag = True
    self_play_count = 10 # for better results

    model = build_model()
    print('aqusiting data using randoem.randrange(0,2)...')
    training_data = data_preparation(N=N, K=K, f=lambda s: random.randrange(0,2), render=True)
    print('training model with training_data....')
    train_model(model=model, training_set=training_data)

    def predictor(s):
        return np.random.choice([0,1], p=model.predict(s.reshape(-1,4))[0])

    if(self_play_flag):
        for i in range(self_play_count):
            K = (N//9+K)//2
            print('iterative data_acquisition with model prediction... {0}/{1}..'.format(i,self_play_count))
            training_data = data_preparation(N=N, K=K, f=predictor, render=True)
            train_model(model=model, training_set=training_data)

    print('final testing...')
    data_preparation(100, 100, predictor, True)