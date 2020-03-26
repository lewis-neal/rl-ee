import numpy as np

class UCB_1:
    def __init__(self, c, state_dim, action_dim, discount_factor):
        self.__c = c
        self.__state_dim = state_dim
        self.__action_dim = action_dim
        self.reset()
        self.__bonus_func = np.vectorize(self.__get_bonus)
        self.__discount_factor = discount_factor

    def __update_count(self, state, action):
        self.__counts[state, action] += 1

    def __get_bonuses(self, state):
        total = sum(self.__counts[state,:])
        return self.__bonus_func(self.__counts[state,:], total)

    def __get_bonus(self, count, total):
        # as per: Comparing Exploration Strategies for Q-Learning in Random Stochastic Mazes
        # Must take each action once upon visiting a state before taking any others
        # so force this by providing a large exploration bonus
        if count == 0:
            return 99999999
        return self.__c * ((2 * np.log(total) / count) ** 0.5)

    def reset(self):
        self.__counts = np.zeros([self.__state_dim, self.__action_dim])

    def select_action(self, state, q_function, env):
        action = q_function.get_best_action(state, self.__get_bonuses(state))
        self.__update_count(state, action)
        return action

    def post_update(self, state, action, td_error):
        self.__update_count(state, action)
