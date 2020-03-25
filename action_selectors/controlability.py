import numpy as np

class Controlability:
    def __init__(self, beta, state_dim, action_dim, learning_rate, omega):
        self.__alpha = beta * learning_rate
        self.__state_dim = state_dim
        self.__action_dim = action_dim
        self.reset()
        self.__omega = omega 

    def __update_bonus(self, state, action, td_error):
        self.__bonuses[state, action] -= self.__alpha * (abs(td_error) + self.__bonuses[state, action])

    def __get_bonuses(self, state):
        return self.__bonuses[state,:] * self.__omega

    def reset(self):
        self.__bonuses = np.zeros([self.__state_dim, self.__action_dim])

    def select_action(self, state, q_function, env):
        action = q_function.get_best_action(state, self.__get_bonuses(state))
        return action

    def post_update(self, state, action, td_error):
        self.__update_bonus(state, action, td_error)
