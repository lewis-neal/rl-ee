import numpy as np

class MBIE_EB:
    def __init__(self, beta, state_dim, action_dim, discount_factor):
        self.__beta = beta
        self.__state_dim = state_dim
        self.__action_dim = action_dim
        self.reset()
        self.__bonus_func = np.vectorize(self.__get_bonus)
        self.__discount_factor = discount_factor

    def __update_count(self, state, action):
        self.__counts[state, action] += 1

    def __get_bonuses(self):
        return self.__bonus_func(self.__counts)

    def __get_bonus(self, count):
        # as per: An Analysis of Model-Based Interval Estimation for Markov Decision Processes
        if count == 0:
            return 1 / (1 - self.__discount_factor)
        return self.__beta * 1/(count ** 0.5) 

    def reset(self):
        self.__counts = np.zeros([self.__state_dim, self.__action_dim])

    def select_action(self, state, q_function):
        action = q_function.get_best_action(state, self.__get_bonuses()) 
        self.__update_count(state, action)
        return action

