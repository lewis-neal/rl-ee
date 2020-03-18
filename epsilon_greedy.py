import numpy as np

class EpsilonGreedy:
    def __init__(self, epsilon, epsilon_discount_factor):
        self.__epsilon = epsilon
        self.__original_epsilon = epsilon
        self.__epsilon_discount_factor = epsilon_discount_factor

    def select_action(self, current_state, q_function, env):
        if np.random.uniform() > self.__epsilon:
            self.__update_epsilon()
            action = q_function.get_best_action(current_state)
            return action
        self.__update_epsilon()
        return env.get_random_action()

    def __update_epsilon(self):
        self.__epsilon *= self.__epsilon_discount_factor

    def reset(self):
        self.__epsilon = self.__original_epsilon
