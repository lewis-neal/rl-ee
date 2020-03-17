import numpy as np

class ActionWrapper:
    def __init__(self, high, low, num_actions):
        self.__high = high
        self.__low = low
        self.__num_actions = num_actions

    def __get_actions(self):
        high = self.__high
        low = self.__low
        num_actions = self.__num_actions - 1
        action_list = []
        num_actions -= 1
        for i in np.arange(low, high + ((high - low) / num_actions), (high - low) / num_actions):
            action_list.append(i)
        return action_list

    def discretise(self, action):
        total = 0
        actions = self.__get_actions()
        disc_action = self.__get_action(action, actions)
        return disc_action

    def __get_action(self, value, action_list):
        low = -9999999999
        high = 9999999999
        low_ind = 0
        high_ind = 0
        for ind, s in enumerate(action_list):
            if value < s:
                high = s
                high_ind = ind
                break
            low = s
            low_ind = ind

        diff_low = abs(value - low)
        diff_high = abs(value - high)

        if diff_low > diff_high:
            return high_ind
        return low_ind

    def undiscretise(self, value):
        low = self.__low
        high = self.__high
        step = (high - low) / self.__num_actions
        return [low + (value * step)]
