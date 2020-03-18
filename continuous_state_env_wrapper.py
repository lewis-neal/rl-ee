import numpy as np

class ContinuousStateEnvWrapper:
    def __init__(self, env, num_states):
        self.__env = env
        self.__num_states = num_states
        self.__states = self.__get_states_list(self.__num_states)

    def reset(self):
        return self.__discretise(self.__env.reset())

    def step(self, action):
        next_state, reward, done, info = self.__env.step(action)
        return self.__discretise(next_state), reward, done, info

    def render(self):
        self.__env.render()

    def close(self):
        self.__env.close()

    def get_random_action(self):
        return self.__env.action_space.sample()

    def get_total_actions(self):
        return self.__env.action_space.n

    def get_total_states(self):
        total = 1
        for dim in self.__num_states:
            if dim == 0:
                continue
            total *= dim
        return total

    def __discretise(self, state):
        total = 0
        for ind, dim in enumerate(self.__num_states):
            if dim == 0:
                continue
            disc_state = self.__get_state(state[ind], self.__states[ind])
            total += self.__get_factor(ind) * disc_state
        return total

    def __get_factor(self, pos):
        if pos == 0:
            return 1
        if pos == 1:
            return self.__num_states[0]
        return self.__num_states[pos-1] * self.__get_factor(pos - 1)

    def __get_states_list(self, num_states):
        states_list = []
        for ind, dim in enumerate(num_states):
            states = self.__get_states(ind, dim)
            states_list.append(states)
        return states_list

    def __get_states(self, ind, dim):
        if dim == 0:
            return []
        high = self.__env.observation_space.high[ind]
        low = self.__env.observation_space.low[ind]
        states = []
        dim -= 1
        for i in np.arange(low, high + ((high - low) / dim), (high - low) / dim):
            states.append(i)
        return states

    def __get_state(self, value, state_list):
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
