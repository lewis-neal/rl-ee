import numpy as np

class PendulumEnvWrapper:
    def __init__(self, env, num_states, num_actions):
        self.__env = env
        self.__num_states = num_states
        self.__num_actions = num_actions
        self.__states = self.__get_values_list(self.__num_states, 'state')
        self.__actions = self.__get_values_list(self.__num_actions, 'action')

    def reset(self):
        return self.__discretise(self.__env.reset(), 'state')

    def step(self, action):
        action = self.__undiscretise(action)
        next_state, reward, done, info = self.__env.step(action)
        return self.__discretise(next_state, 'state'), reward, done, info

    def render(self):
        self.__env.render()

    def close(self):
        self.__env.close()

    def get_random_action(self):
        return self.__discretise(self.__env.action_space.sample(), 'action')

    def get_total_actions(self):
        total = 1
        for dim in self.__num_actions:
            if dim == 0:
                continue
            total *= dim
        return total


    def get_total_states(self):
        total = 1
        for dim in self.__num_states:
            if dim == 0:
                continue
            total *= dim
        return total

    def __discretise(self, value, val_type):
        total = 0
        if val_type == 'state':
            vals = self.__num_states
            val_list = self.__states
        else:
            vals = self.__num_actions
            val_list = self.__actions
        for ind, dim in enumerate(vals):
            if dim == 0:
                continue
            disc_val = self.__get_val(value[ind], val_list[ind])
            total += self.__get_factor(ind) * disc_val
        return total

    def __get_factor(self, pos):
        if pos == 0:
            return 1
        if pos == 1:
            return self.__num_states[0]
        return self.__num_states[pos-1] * self.__get_factor(pos - 1)

    def __get_values_list(self, num_values, val_type):
        values_list = []
        for ind, dim in enumerate(num_values):
            values = self.__get_values(ind, dim, val_type)
            values_list.append(values)
        return values_list

    def __get_values(self, ind, dim, val_type):
        if dim == 0:
            return []
        if val_type == 'state':
            high = self.__env.observation_space.high[ind]
            low = self.__env.observation_space.low[ind]
        else:
            high = self.__env.action_space.high[ind]
            low = self.__env.action_space.low[ind]
        values = []
        dim -= 1
        for i in np.arange(low, high + ((high - low) / dim), (high - low) / dim):
            values.append(i)
        return values

    def __get_val(self, value, val_list):
        low = -9999999999
        high = 9999999999
        low_ind = 0
        high_ind = 0
        for ind, s in enumerate(val_list):
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

    def __undiscretise(self, value):
        high = self.__env.action_space.high[0]
        low = self.__env.action_space.low[0]
        step = (high - low) / self.__num_actions[0]
        return [low + (value * step)]

    def seed(self, seed=None, set_action_seed=True):
        if set_action_seed:
            self.__env.action_space.seed(seed)
        return self.__env.seed(seed)
