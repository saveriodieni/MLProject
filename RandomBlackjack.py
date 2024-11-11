import gymnasium as gym
import numpy as np

# Define the environment
env = gym.make('Blackjack-v1')

for iteration in range(100):

   test_episodes = 100000

   total_reward=0
   for i in range(test_episodes):
      observation, info = env.reset()
      done = False
      episode_reward = 0

      while not done:
         action = env.action_space.sample()

         # Take action and observe the result
         observation, reward, terminated, truncated, info = env.step(action)
         done = terminated or truncated
         episode_reward += reward
      total_reward+=episode_reward

   total_reward /= test_episodes
   print(f"Iteration: {iteration+1} , Total Reward: {total_reward}")

env.close()