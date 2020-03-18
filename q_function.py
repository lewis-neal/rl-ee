import numpy as np

class Q:
    def __init__(self, observation_count, action_count, learning_rate, discount_factor):
        self.__observation_count = observation_count 
        self.__action_count = action_count
        self.__learning_rate = learning_rate
        self.__discount_factor = discount_factor
        self.reset()

    def get_q_function(self):
        return self.__q_function

    def update_q_function(self, current_state, next_state, action, reward):
        self.__q_function[current_state, action] = self.__q_function[current_state, action] + self.__learning_rate *\
                (reward + self.__discount_factor * np.max(self.__q_function[next_state, :]) - self.__q_function[current_state, action])

    def get_best_action(self, state, bonuses=[]):
        temp_q = self.__q_function
        if len(bonuses) > 0:
            temp_q = np.add(temp_q, bonuses)
        return np.argmax(temp_q[state,:])

    def reset(self):
        self.__q_function = np.zeros([self.__observation_count, self.__action_count])

    def get_actions_for_state(self, state):
        return self.__q_function[state,:]
