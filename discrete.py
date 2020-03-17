import numpy as np

class Discrete:
    def __init__(self, num_states, env):
        self.__num_states = num_states
        self.__env = env

    def get_states(self, dim, num_states):
        high = self.__env.observation_space.high[dim]
        low = self.__env.observation_space.low[dim]
        state_list = []
        num_states -= 1
        for i in np.arange(low, high + ((high - low) / num_states), (high - low) / num_states):
            state_list.append(i)
        return state_list

    def discretise(self, state):
        total = 0
        for i in range(len(self.__num_states)):
            if self.__num_states[i] == 0:
                continue
            states = self.get_states(i, self.__num_states[i])
            disc_state = self.get_state(state[i], states)
            total += (self.__get_factor(i) * disc_state)
        return total

    def __get_factor(self, i):
        if i == 0:
            return 1
        if i == 1:
            return self.__num_states[0]
        return (self.__num_states[i-1] + 1) * self.__get_factor(i-1)

    def get_state(self, value, state_list):
        low = -9999999999
        high = 9999999999
        low_ind = 0
        high_ind = 0
        for ind, s in enumerate(state_list):
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

    def get_total_states(self):
        total = 1
        for num in self.__num_states:
            if num == 0:
                continue
            total *= num

        return total
