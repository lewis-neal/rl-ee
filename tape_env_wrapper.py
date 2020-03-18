class TapeEnvWrapper:
    def __init__(self, env):
        self.__env = env
        self.__factors = [1, 2, 4]

    def reset(self):
        return self.__env.reset()

    def step(self, action):
        action = self.__undiscretise(action)
        next_state, reward, done, info = self.__env.step(action)
        return next_state, reward, done, info

    def render(self):
        self.__env.render()

    def close(self):
        self.__env.close()

    def get_random_action(self):
        action = self.__env.action_space.sample()
        total = 0
        pointer = 0
        for dim in action:
           total += dim * self.__factors[pointer]
           pointer += 1
        return total

    def get_total_actions(self):
        total = 1
        for dim in self.__env.action_space:
            total *= dim.n
        return total

    def get_total_states(self):
        return self.__env.observation_space.n

    def __undiscretise(self, action):
        act = [0, 0, 0]
        for i in range(2, -1, -1):
            act[i] = action // self.__factors[i]
            action = action % self.__factors[i]
        return tuple(act)
