import numpy as np
import datetime

class Logger:
    def __init__(self, episodes, iterations):
        self.__logs = np.zeros([episodes * iterations, 5])
        self.__episodes = episodes
        self.__iterations = iterations
        self.__pointer = 0

    def log(self, iteration, episode, reward, total_reward, episode_length):
        self.__logs[self.__pointer][0] = iteration
        self.__logs[self.__pointer][1] = episode
        self.__logs[self.__pointer][2] = reward
        self.__logs[self.__pointer][3] = total_reward
        self.__logs[self.__pointer][4] = episode_length
        self.__pointer += 1

    def write(self, file_name, data=[]):
        if len(data) == 0:
            data = self.__logs
        np.savetxt(file_name, data, delimiter=',')
