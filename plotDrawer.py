import os
import matplotlib.pyplot as plt

def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    train_episodes = []
    average_rewards = []
    
    for line in lines:
        parts = line.strip().split(',')
        episode = int(parts[0].split(': ')[1])
        reward = float(parts[1].split(': ')[1])
        train_episodes.append(episode)
        average_rewards.append(reward)
    
    return train_episodes, average_rewards

def plot_data(train_episodes, average_rewards, file_path,threshold):
    plt.figure(figsize=(10, 6))
    plt.plot(train_episodes, average_rewards, marker='o', linestyle='-', color='b')

    # Add dashed orange line at the threshold value
    if threshold is not None:
        plt.axhline(y=threshold, color='orange', linestyle='--', label=f'{threshold}')

    plt.title('Average Reward vs. Train Episodes')
    plt.xlabel('Train Episodes')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    # Extract the base name of the file without extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Save the plot in the same directory as the file_path
    dir_path = os.path.dirname(file_path)
    plot_file_path = os.path.join(dir_path, f'{base_name}.png')
    plt.savefig(plot_file_path)
    
    # Create plots directory if it doesn't exist
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Save the plot in the plots directory
    plots_file_path = os.path.join(plots_dir, f'{base_name}.png')
    plt.savefig(plots_file_path)
    
    # Show the plot
    plt.show()

file_path = 'results-202409142229-sigmoid/results_1000train_episodes_16batch_size_100000deque_maxlen_0.001learning_rate_0.995eps_decay_0.9gamma_sigmoid.txt'

# Read data from the file
train_episodes, average_rewards = read_data_from_file(file_path)

# Generate the plot and save it in the desired locations
plot_data(train_episodes, average_rewards, file_path,None)