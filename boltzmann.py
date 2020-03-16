import numpy as np
from math import exp

class Boltzmann:
    def __init__(self, temperature):
        self.__temperature = temperature
        self.__original_temp = temperature

    def select_action(self, state, q_function, env):
        probabilities = self.__get_action_probabilities(state, q_function)
        self.__update_temperature()
        num = np.random.uniform()
        prob_total = 0
        for action in probabilities:
            prob_total += action[0]
            if num < prob_total:
                return int(action[1])

    def __get_action_probabilities(self, state, q_function):
        means, total = self.__get_means(state, q_function)   
        actions = []
        for value in means:
            action = np.zeros(2)
            prob = value[0] / total
            action[0] = prob
            action[1] = value[1]
            actions.append(action)
        return actions



    def __get_means(self, state, q_function):
        means = []
        q_values = q_function.get_actions_for_state(state)
        q_values = np.sort(q_values)[::-1]
        total = 0
        pointer = 0
        for value in q_values:
            action = np.zeros(2)
            mean = self.__get_mean(value)
            total += mean
            action[0] = mean
            action[1] = pointer
            means.append(action)
            pointer += 1
        return means, total

    def __get_mean(self, q_value):
        return exp(q_value / self.__temperature)

    def __update_temperature(self):
        if self.__temperature == 1:
            return
        self.__temperature -= 1

    def reset(self):
       self.__temperature = self.__original_temp 
