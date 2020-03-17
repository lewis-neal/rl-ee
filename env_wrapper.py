class EnvWrapper:
    def __init__(self, env, num_states=[], discrete=None):
        self.__env = env
        self.action_space = self.__env.action_space
        self.observation_space = self.__env.observation_space
        self.__discrete_states = not (len(num_states) > 0)
        if not self.__discrete_states:
            self.__discrete = discrete

    def reset(self):
        if self.__discrete_states:
            return self.__env.reset()
        return self.__discrete.discretise(self.__env.reset())

    def step(self, action):
        next_state, reward, done, info = self.__env.step(action)
        if not self.__discrete_states:
            next_state = self.__discrete.discretise(next_state)
        return next_state, reward, done, info

    def render(self):
        self.__env.render()

    def close(self):
        self.__env.close()
