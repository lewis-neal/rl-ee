import numpy as np

class EpsilonGreedy:
    def __init__(self, epsilon, epsilon_discount_factor, seed=None):
        self.__epsilon = epsilon
        self.__original_epsilon = epsilon
        self.__epsilon_discount_factor = epsilon_discount_factor
        if not seed == None:
            np.random.seed(seed)

    def select_action(self, current_state, q_function, env):
        if np.random.uniform() > self.__epsilon:
            action = q_function.get_best_action(current_state)
            return action
        return env.get_random_action()

    def __update_epsilon(self):
        self.__epsilon *= self.__epsilon_discount_factor

    def reset(self):
        self.__epsilon = self.__original_epsilon

    def post_update(self, state, action, td_error):
        self.__update_epsilon()
