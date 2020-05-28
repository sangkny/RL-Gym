# import librarys

import gym
import random

env = gym.make('CartPole-v1')    # make an envirenment named 'Cartpole-v1' which has been defined in the library
goal_steps = 100

while True:
    obs = env.reset() # reset the status
    for i in range(goal_steps):
        obs, reward, done, info = env.step(random.randrange(0,2)) # set the random number which is the maximum numbers we can choose to make the bar uphold
        # obs: object current status, reward: reward according to the performance, done: bar is out of range or falled down
        if done:
            break # escape the while loop if done

        env.render() # render and display the environment
