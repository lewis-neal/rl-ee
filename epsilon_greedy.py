import numpy as np
from action_space_wrapper import ActionWrapper

class EpsilonGreedy:
    def __init__(self, epsilon, epsilon_discount_factor, action_wrapper=None):
        self.__epsilon = epsilon
        self.__original_epsilon = epsilon
        self.__epsilon_discount_factor = epsilon_discount_factor
        self.__discretise = action_wrapper == None
        if not self.__discretise:
            self.__action_wrapper = action_wrapper

    def select_action(self, current_state, q_function, env):
        if np.random.uniform() > self.__epsilon:
            self.__update_epsilon()
            action = q_function.get_best_action(current_state)
            if not self.__discretise:
                return self.__action_wrapper.undiscretise(action)
            return action
        self.__update_epsilon()
        return env.action_space.sample()

    def __update_epsilon(self):
        self.__epsilon *= self.__epsilon_discount_factor

    def reset(self):
        self.__epsilon = self.__original_epsilon
