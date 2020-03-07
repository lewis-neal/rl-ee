import numpy as np
import datetime

class Logger:
    def __init__(self, episodes, file_name, iterations):
        self.__logs = np.zeros([episodes, 4])
        self.__episodes = episodes
        self.__iterations = iterations
        self.__file_name = file_name
        self.__done_func = np.vectorize(self.__average)

    def __average(self, reward):
        return reward / self.__iterations

    def log(self, episode, reward, iteration, total_reward):
        self.__logs[episode - 1][0] += iteration
        self.__logs[episode - 1][1] += episode
        self.__logs[episode - 1][2] += reward
        self.__logs[episode - 1][3] += total_reward

    def __done(self):
        self.__logs = self.__done_func(self.__logs)

    def write(self):
        self.__done()
        print(self.__logs)
        np.savetxt(self.__file_name, self.__logs, delimiter=',')
