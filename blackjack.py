import gym
env = gym.make('Blackjack-v0')
print(env.observation_space)
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        #env.render()
        print(observation)
        action = env.action_space.sample()
        if action == 1:
            print('Hit')
        else:
            print('Stick')
        observation, reward, done, info = env.step(action)
        if reward == -1:
            print('Lost')
        elif reward == 0:
            print('Drew')
        else:
            print('Won')
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
