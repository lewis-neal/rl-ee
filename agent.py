class Agent:
    def __init__(self, env, q_function, action_selector, logger):
        self.__env = env
        self.__q_function = q_function
        self.__action_selector = action_selector
        self.__logger = logger

    def train(self, steps, episodes):
        episode_reward = 0
        total_reward = 0
        self.__q_function.reset()
        self.__action_selector.reset()
        for episode in range(episodes):
            current_state = self.__env.reset()
            episode_reward = 0
            episode_length = 0
            for t in range(steps):
                action = self.__action_selector.select_action(current_state, self.__q_function, self.__env)
                next_state, reward, done, info = self.__env.step(action)
                self.__q_function.update_q_function(current_state, next_state, action, reward)
                current_state = next_state
                episode_reward += reward
                episode_length += 1
                if done:
                    break
            total_reward += episode_reward
            self.__logger.log(episode, episode_reward, total_reward, episode_length)

    def solve(self, steps, filepath, render):
        episode_reward = 0
        current_state = self.__env.reset()
        if render:
            self.__env.render()
        for i in range(steps):
            action = self.__q_function.get_best_action(current_state)
            next_state, reward, done, info = self.__env.step(action)
            current_state = next_state
            episode_reward += reward
            if render:
                self.__env.render()
            if done:
                break
        print("Episode finished after {} timesteps".format(i+1))
        print("Cumulative reward at end = " + str(episode_reward))
        self.__logger.write(filepath + '.csv')
        self.__logger.write(filepath + '-q-function.csv', self.__q_function.get_q_function())

