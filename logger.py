import numpy as np
import datetime

class Logger:
    def __init__(self, episodes, log_dir, file_name):
        self.__logs = np.zeros(episodes)
        self.__episodes = episodes
        self.__log_dir = log_dir
        self.__file_name = file_name
        self.__done_func = np.vectorize(self.__average)

    def __average(self, reward):
        return reward / self.__episodes

    def log(self, episode, reward):
        self.__logs[episode - 1] += reward

    def __done(self):
        self.__logs = self.__done_func(self.__logs)

    def write(self):
        self.__done()
        np.savetxt(self.__log_dir + self.__file_name, self.__logs, delimiter=',')
