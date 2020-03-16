import gym

class EnvWrapper:
    def __init__(self, name):
        self.__env = gym.make(name)
        self.action_space = self.__env.action_space
        self.observation_space = self.__env.observation_space

    def reset(self):
        return self.__env.reset()

    def step(self, action):
        next_state, reward, done, info = self.__env.step(action)
        return next_state, reward, done, info

    def render(self):
        self.__env.render()

