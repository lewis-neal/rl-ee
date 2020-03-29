class DiscreteEnvWrapper:
    def __init__(self, env):
        self.__env = env

    def reset(self):
        return self.__env.reset()

    def step(self, action):
        next_state, reward, done, info = self.__env.step(action)
        return next_state, reward, done, info

    def render(self):
        self.__env.render()

    def close(self):
        self.__env.close()

    def get_random_action(self):
        return self.__env.action_space.sample()

    def get_total_actions(self):
        return self.__env.action_space.n

    def get_total_states(self):
        return self.__env.observation_space.n

    def seed(self, seed=None):
        self.__env.action_space.seed(seed)
        return self.__env.seed(seed)
