import numpy as np
from math import exp

class VDBE:
    def __init__(self, state_dim, delta, inv_sens, learning_rate):
        self.__state_dim = state_dim
        self.__delta = delta
        # inv sens must be > 0
        self.__inv_sens = inv_sens
        self.__learning_rate = learning_rate
        self.reset()

    def __update_epsilon(self, state, td_error):
        self.__epsilons[state] = (self.__delta * self.__calculate_f(state, td_error)) + ((1 - self.__delta) * self.__epsilons[state])

    def __get_epsilon(self, state):
        return self.__epsilons[state]

    def reset(self):
        self.__epsilons = np.ones(self.__state_dim)

    def select_action(self, state, q_function, env):
        epsilon = self.__get_epsilon(state)
        if np.random.uniform() > epsilon:
            action = q_function.get_best_action(state)
            return action
        return env.get_random_action()

    def post_update(self, state, action, td_error):
        self.__update_epsilon(state, td_error)

    def __calculate_f(self, state, td_error):
        boltz = (-abs(self.__learning_rate * td_error)) / self.__inv_sens
        return (1 - exp(boltz)) / (1 + exp(boltz))
