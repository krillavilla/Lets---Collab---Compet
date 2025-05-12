import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import csv
import time
from unityagents import UnityEnvironment
from model import MultiAgentDDPG, RewardNormalizer

def train(env, agent, n_episodes=3000, max_t=1000, print_every=10, checkpoint_every=500, 
         solve_score=0.5, noise_decay_rate=0.995, noise_decay_every=100,
         lr_step_every=100, normalize_rewards=True):
    """Train the agent.

    Params
    ======
        env: Unity environment
        agent: MultiAgentDDPG agent
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        print_every (int): print average score every print_every episodes
        checkpoint_every (int): save checkpoint every checkpoint_every episodes
        solve_score (float): score to consider the environment solved
        noise_decay_rate (float): rate at which to decay exploration noise
        noise_decay_every (int): decay noise every noise_decay_every episodes
        lr_step_every (int): step learning rate scheduler every lr_step_every episodes
        normalize_rewards (bool): whether to normalize rewards
    """
    brain_name = env.brain_names[0]
    scores = []
    scores_window = []  # last 100 scores
    best_score = -np.inf

    # Create reward normalizers for each agent if needed
    reward_normalizers = [RewardNormalizer() for _ in range(len(agent.agents))] if normalize_rewards else None

    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Create a CSV file for logging
    log_file = f'logs/training_log_{time.strftime("%Y%m%d-%H%M%S")}.csv'
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Score', 'Avg_Score', 'Actor_LR', 'Critic_LR', 'Noise_Sigma', 'Critic_Loss', 'Actor_Loss'])

    # Track training speed
    start_time = time.time()
    steps_done = 0

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros(len(env_info.agents))
        episode_steps = 0

        # Initialize loss tracking variables
        critic_losses = []
        actor_losses = []

        # Log episode start
        print(f'\rEpisode {i_episode}/{n_episodes}', end="")

        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # Normalize rewards if needed
            if normalize_rewards:
                normalized_rewards = [normalizer.normalize(reward) for normalizer, reward in zip(reward_normalizers, rewards)]
                losses = agent.step(states, actions, normalized_rewards, next_states, dones)
            else:
                losses = agent.step(states, actions, rewards, next_states, dones)

            states = next_states
            score += rewards
            episode_steps += 1
            steps_done += 1

            # Track losses for logging (if available)
            if losses and losses[0] is not None:  # Check if we have losses from the first agent
                critic_losses.append(losses[0][0])  # Critic loss from first agent
                actor_losses.append(losses[0][1])   # Actor loss from first agent

            if np.any(dones):
                break

        # Get the maximum score between the two agents
        episode_score = np.max(score)
        scores.append(episode_score)
        scores_window.append(episode_score)

        # Keep only the last 100 scores in the window
        if len(scores_window) > 100:
            scores_window.pop(0)

        # Decay noise periodically to reduce exploration over time
        if i_episode % noise_decay_every == 0:
            agent.decay_noise(noise_decay_rate)
            print(f'\rNoise decayed at episode {i_episode}')

        # Step learning rate schedulers periodically
        if i_episode % lr_step_every == 0:
            agent.step_schedulers()
            print(f'\rLearning rates updated at episode {i_episode}')

        # Calculate training speed
        elapsed_time = time.time() - start_time
        steps_per_second = steps_done / elapsed_time if elapsed_time > 0 else 0

        # Get current learning rates and noise level
        lr_info = agent.get_lr()[0]  # Get info from first agent
        noise_sigma = agent.agents[0].noise.sigma

        # Calculate average losses if available
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0.0

        # Log to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i_episode, episode_score, np.mean(scores_window), 
                            lr_info['actor_lr'], lr_info['critic_lr'], noise_sigma,
                            avg_critic_loss, avg_actor_loss])

        # Print progress
        if i_episode % print_every == 0:
            avg_score = np.mean(scores_window)
            print(f'\rEpisode {i_episode}\tAverage Score: {avg_score:.3f}\tScore: {episode_score:.3f}\tSteps/sec: {steps_per_second:.2f}')
            print(f'\rActor LR: {lr_info["actor_lr"]:.6f}\tCritic LR: {lr_info["critic_lr"]:.6f}\tNoise Sigma: {noise_sigma:.6f}')
            print(f'\rCritic Loss: {avg_critic_loss:.6f}\tActor Loss: {avg_actor_loss:.6f}')

            # Save checkpoint if we have a new best average score
            if avg_score > best_score and len(scores_window) >= 100:
                best_score = avg_score
                agent.save_checkpoint(f'checkpoints/tennis_best_checkpoint.pth')
                print(f'\rNew best average score: {best_score:.3f}. Checkpoint saved.')

        # Save checkpoint periodically
        if i_episode % checkpoint_every == 0:
            agent.save_checkpoint(f'checkpoints/tennis_checkpoint_{i_episode}.pth')
            print(f'\rCheckpoint saved at episode {i_episode}')

            # Also save a plot of the scores so far
            plot_scores(scores, filename=f'logs/scores_episode_{i_episode}.png', window_size=100)

        # Check if the environment is solved
        if i_episode >= 100 and np.mean(scores_window) >= solve_score:
            print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.3f}')
            agent.save_checkpoint('checkpoints/tennis_solved_checkpoint.pth')

            # Save the final model
            agent.save_checkpoint('checkpoints/tennis_final_checkpoint.pth')
            break

    # Save the final model if not solved
    if np.mean(scores_window) < solve_score:
        agent.save_checkpoint('checkpoints/tennis_final_checkpoint.pth')
        print(f'\nTraining completed without solving the environment. Final Average Score: {np.mean(scores_window):.3f}')

    return scores

def plot_scores(scores, filename='scores.png', window_size=100):
    """Plot the scores and moving average.

    Params
    ======
        scores (list): List of scores
        filename (str): Filename to save the plot
        window_size (int): Size of the window for moving average
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Plot raw scores
    x = np.arange(len(scores))
    y = np.array(scores)
    plt.plot(x, y, alpha=0.3, label='Raw Scores')

    # Plot moving average if we have enough data
    if len(scores) >= window_size:
        moving_avg = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(scores)), moving_avg, label=f'Moving Avg (window={window_size})')

    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.title('Training Scores')
    plt.grid(True)
    plt.savefig(filename)

    print(f"Plot saved to {filename}")

if __name__ == '__main__':
    # Load the environment in headless mode for faster training
    env = UnityEnvironment(file_name="env/Tennis_Linux/Tennis.x86_64", no_graphics=True)

    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # Get the number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # Get the size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # Get the state size
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))

    # Define agent hyperparameters
    agent_params = {
        'buffer_size': int(1e5),   # Replay buffer size
        'batch_size': 64,          # Minibatch size (smaller for more frequent updates)
        'gamma': 0.99,             # Discount factor (increased for longer-term rewards)
        'tau': 1e-3,               # For soft update of target parameters
        'lr_actor': 1e-3,          # Learning rate for actor
        'lr_critic': 1e-3,         # Learning rate for critic (increased for faster learning)
        'weight_decay': 1e-4,      # L2 weight decay for critic (added to prevent overfitting)
        'network_size': 'large',   # Network size ('small' or 'large') - using larger network
        'activation_fn': 'leaky_relu',  # Activation function ('relu', 'leaky_relu', or 'elu')
        'use_batch_norm': True,    # Whether to use batch normalization
        'grad_clip': 1.0           # Value for gradient clipping
    }

    # Create the agent with a random seed for reproducibility
    random_seed = 42
    agent = MultiAgentDDPG(state_size, action_size, num_agents, random_seed=random_seed, **agent_params)

    # Define training hyperparameters
    training_params = {
        'n_episodes': 5000,        # Maximum number of training episodes (increased for more learning time)
        'max_t': 1000,             # Maximum timesteps per episode
        'print_every': 10,         # Print progress every N episodes
        'checkpoint_every': 200,   # Save checkpoint more frequently
        'solve_score': 0.5,        # Score to consider the environment solved
        'noise_decay_rate': 0.995, # Slower noise decay to maintain exploration longer
        'noise_decay_every': 10,   # Decay noise less frequently
        'lr_step_every': 200,      # Step learning rate scheduler less frequently
        'normalize_rewards': True  # Whether to normalize rewards
    }

    print(f"Starting training with seed {random_seed}")
    print("Agent parameters:")
    for param, value in agent_params.items():
        print(f"  {param}: {value}")
    print("Training parameters:")
    for param, value in training_params.items():
        print(f"  {param}: {value}")

    # Train the agent with the specified parameters
    scores = train(env, agent, **training_params)

    # Plot the scores with a moving average
    plot_scores(scores, filename='logs/final_scores.png', window_size=100)

    # Close the environment
    env.close()
