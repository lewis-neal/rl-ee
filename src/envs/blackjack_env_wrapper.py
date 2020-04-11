class BlackjackEnvWrapper:
    def __init__(self, env):
        self.__env = env
        self.__factors = self.__get_factors()

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
        for dim in self.__env.observation_space:
            total *= dim.n
        return total

    def __discretise(self, state):
        total = 0
        pointer = 0
        for dim in state:
            total += dim * self.__factors[pointer]
            pointer += 1
        return total

    def __get_factors(self):
        factors = []
        for i in range(len(self.__env.observation_space)):
           factors.append(self.__get_factor(i, self.__env.observation_space))
        return factors

    def __get_factor(self, pos, observation_space):
        if pos == 0:
            return 1
        if pos == 1:
            return observation_space[0].n
        return observation_space[pos-1].n * self.__get_factor(pos - 1, observation_space)

    def seed(self, seed=None, set_action_seed=True):
        if set_action_seed:
            self.__env.action_space.seed(seed)
        return self.__env.seed(seed)
