import gymnasium as gym
import numpy as np
from collections import defaultdict

# Define parameters
train_episodes = 100000
test_episodes = 100000
learning_rate = 0.001
discount_factor = 0.9
exploration_prob = 1.0
exploration_decay = 0.999
min_exploration_prob = 0.01

# Define the environment
env = gym.make('Blackjack-v1')

# Initialize Q-table as a dictionary
Q_table = defaultdict(lambda: np.zeros(env.action_space.n))

def get_state(observation):
   """Convert observation tuple to a single state number"""
   player_sum, dealer_card, usable_ace = observation
   return (player_sum, dealer_card, usable_ace)

def test():
   # Test the learned policy
   total_reward = 0
   for i in range(test_episodes):
      observation, info = env.reset()
      done = False
      episode_reward = 0
      current_state = get_state(observation)

      while not done:
         action = np.argmax(Q_table[current_state])

         # Take action and observe the result
         observation, reward, terminated, truncated, info = env.step(action)
         done = terminated or truncated
         next_state = get_state(observation)

         episode_reward += reward
         current_state = next_state
      total_reward += episode_reward

   avg_reward = total_reward / test_episodes
   print(f"Final Average Reward: {avg_reward}")
   env.close()

# Training phase
cumulative_reward = 0
for episode in range(train_episodes):
   observation, info = env.reset()
   done = False
   episode_reward = 0
   current_state = get_state(observation)

   while not done:
      # Epsilon-greedy action selection
      if np.random.rand() < exploration_prob:
         action = env.action_space.sample()
      else:
         action = np.argmax(Q_table[current_state])

      # Take action and observe the result
      observation, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
      next_state = get_state(observation)

      # Q-learning update
      best_next_action = np.argmax(Q_table[next_state])
      Q_table[current_state][action] += learning_rate * (
          reward + discount_factor * Q_table[next_state][best_next_action] - Q_table[current_state][action])

      episode_reward += reward
      current_state = next_state

   cumulative_reward += episode_reward

   # Decrease exploration probability
   exploration_prob = max(min_exploration_prob, exploration_prob * exploration_decay)
   if (episode + 1) % 1000 == 0:
      avg_reward = cumulative_reward / (episode + 1)
      print(f"Train episodes: {episode + 1}, Average Reward: {avg_reward}")

env.close()
print("Learned Q-table:")
for state, actions in Q_table.items():
    print(f"{state}: {actions}")
test()

