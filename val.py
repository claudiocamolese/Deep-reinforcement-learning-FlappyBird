import torch
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt

# Use 'Agg' backend for matplotlib when not rendering
matplotlib.use('Agg')

# Date format for logging
DATE_FORMAT = "%m-%d %H:%M:%S"


def validate_agent(agent, render=False, num_episodes=10, max_steps_per_episode=None):
    """
    Validate/test the trained DQN agent.
    
    Args:
        agent (Agent): The agent to validate
        render (bool): Whether to render the environment
        num_episodes (int): Number of episodes to run for validation
        max_steps_per_episode (int): Maximum steps per episode (None for unlimited)
    
    Returns:
        dict: Validation results containing statistics
    """
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Starting validation...")
    
    # Setup environment and load trained model
    agent.setup_environment(render=render)
    agent.setup_networks(for_training=False)
    
    # Validation tracking
    episode_rewards = []
    episode_lengths = []
    successful_episodes = 0
    
    try:
        for episode in range(num_episodes):
            episode_reward, episode_length, success = _run_validation_episode(
                agent, episode, max_steps_per_episode
            )
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if success:
                successful_episodes += 1
            
            # Log episode results
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Length={episode_length}, "
                  f"Success={'Yes' if success else 'No'}")
    
    except KeyboardInterrupt:
        print("Validation interrupted by user")
        num_episodes = len(episode_rewards)
    
    finally:
        agent.cleanup()
    
    # Calculate statistics
    results = _calculate_validation_stats(
        episode_rewards, episode_lengths, successful_episodes, num_episodes
    )
    
    # Print summary
    _print_validation_summary(results)
    
    # Save validation results
    _save_validation_results(agent, results)
    
    return results


def _run_validation_episode(agent, episode, max_steps_per_episode):
    """
    Run a single validation episode.
    
    Args:
        agent (Agent): The agent
        episode (int): Episode number
        max_steps_per_episode (int): Maximum steps per episode
    
    Returns:
        tuple: (episode_reward, episode_length, success)
    """
    # Reset environment
    state, _ = agent.env.reset()
    state = torch.tensor(state, dtype=torch.float, device=agent.device)
    
    episode_reward = 0.0
    episode_length = 0
    terminated = False
    truncated = False
    
    # Run episode with greedy policy (no exploration)
    while not terminated and not truncated:
        # Check step limit
        if max_steps_per_episode and episode_length >= max_steps_per_episode:
            break
        
        # Select best action (greedy policy)
        action = agent.select_action(state, epsilon=0.0)
        
        # Execute action
        new_state, reward, terminated, truncated, info = agent.env.step(action.item())
        episode_reward += reward
        episode_length += 1
        
        # Move to next state
        state = torch.tensor(new_state, dtype=torch.float, device=agent.device)
    
    # Determine success based on environment or reward threshold
    success = _determine_episode_success(agent, episode_reward, terminated, info)
    
    return episode_reward, episode_length, success


def _determine_episode_success(agent, episode_reward, terminated, info):
    """
    Determine if an episode was successful.
    
    Args:
        agent (Agent): The agent
        episode_reward (float): Total episode reward
        terminated (bool): Whether episode terminated
        info (dict): Environment info
    
    Returns:
        bool: Whether the episode was successful
    """
    # Basic success criteria - can be customized per environment
    if hasattr(agent, 'success_threshold'):
        return episode_reward >= agent.success_threshold
    
    # Default success criteria based on common environments
    env_id = agent.env_id.lower()
    
    if 'cartpole' in env_id:
        # CartPole: success if episode lasted long enough
        return episode_reward >= 195
    elif 'lunar' in env_id:
        # LunarLander: success if positive reward and terminated properly
        return episode_reward >= 200 and terminated
    elif 'flappy' in env_id:
        # FlappyBird: success based on reward threshold
        return episode_reward >= 10
    else:
        # Generic: success if reward is positive and above threshold
        return episode_reward > 0


def _calculate_validation_stats(episode_rewards, episode_lengths, successful_episodes, num_episodes):
    """
    Calculate validation statistics.
    
    Args:
        episode_rewards (list): List of episode rewards
        episode_lengths (list): List of episode lengths
        successful_episodes (int): Number of successful episodes
        num_episodes (int): Total number of episodes run
    
    Returns:
        dict: Dictionary containing all statistics
    """
    if not episode_rewards:
        return {
            'num_episodes': 0,
            'mean_reward': 0,
            'std_reward': 0,
            'min_reward': 0,
            'max_reward': 0,
            'mean_length': 0,
            'std_length': 0,
            'min_length': 0,
            'max_length': 0,
            'success_rate': 0,
            'episode_rewards': [],
            'episode_lengths': []
        }
    
    rewards_array = np.array(episode_rewards)
    lengths_array = np.array(episode_lengths)
    
    return {
        'num_episodes': num_episodes,
        'mean_reward': np.mean(rewards_array),
        'std_reward': np.std(rewards_array),
        'min_reward': np.min(rewards_array),
        'max_reward': np.max(rewards_array),
        'mean_length': np.mean(lengths_array),
        'std_length': np.std(lengths_array),
        'min_length': np.min(lengths_array),
        'max_length': np.max(lengths_array),
        'success_rate': successful_episodes / num_episodes * 100,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


def _print_validation_summary(results):
    """
    Print validation summary statistics.
    
    Args:
        results (dict): Validation results
    """
    print(f"\n{datetime.now().strftime(DATE_FORMAT)}: Validation Summary")
    print("=" * 50)
    print(f"Episodes run: {results['num_episodes']}")
    print(f"Success rate: {results['success_rate']:.1f}%")
    print(f"\nReward Statistics:")
    print(f"  Mean: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
    print(f"\nEpisode Length Statistics:")
    print(f"  Mean: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"  Range: [{results['min_length']}, {results['max_length']}]")
    print("=" * 50)


def _save_validation_results(agent, results):
    """
    Save validation results to file and create visualization.
    
    Args:
        agent (Agent): The agent
        results (dict): Validation results
    """
    # Save results to log file
    log_file = agent.log_file.replace('.log', '_validation.log')
    
    with open(log_file, 'w') as f:
        f.write(f"Validation Results - {datetime.now().strftime(DATE_FORMAT)}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Episodes: {results['num_episodes']}\n")
        f.write(f"Success Rate: {results['success_rate']:.1f}%\n")
        f.write(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}\n")
        f.write(f"Reward Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]\n")
        f.write(f"Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}\n")
        f.write(f"Length Range: [{results['min_length']}, {results['max_length']}]\n")
        f.write("\nEpisode Details:\n")
        
        for i, (reward, length) in enumerate(zip(results['episode_rewards'], results['episode_lengths'])):
            f.write(f"Episode {i+1}: Reward={reward:.2f}, Length={length}\n")
    
    # Create validation plots
    _create_validation_plots(agent, results)


def _create_validation_plots(agent, results):
    """
    Create validation plots showing episode performance.
    
    Args:
        agent (Agent): The agent
        results (dict): Validation results
    """
    if results['num_episodes'] == 0:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    episodes = range(1, results['num_episodes'] + 1)
    
    # Episode rewards
    ax1.plot(episodes, results['episode_rewards'], 'b-o', markersize=4)
    ax1.axhline(y=results['mean_reward'], color='r', linestyle='--', alpha=0.7, label=f'Mean: {results["mean_reward"]:.2f}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Episode lengths
    ax2.plot(episodes, results['episode_lengths'], 'g-o', markersize=4)
    ax2.axhline(y=results['mean_length'], color='r', linestyle='--', alpha=0.7, label=f'Mean: {results["mean_length"]:.1f}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Length (Steps)')
    ax2.set_title('Episode Lengths')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Reward histogram
    ax3.hist(results['episode_rewards'], bins=min(10, results['num_episodes']), alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(x=results['mean_reward'], color='r', linestyle='--', label=f'Mean: {results["mean_reward"]:.2f}')
    ax3.set_xlabel('Reward')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Reward Distribution')
    ax3.legend()
    
    # Length histogram
    ax4.hist(results['episode_lengths'], bins=min(10, results['num_episodes']), alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(x=results['mean_length'], color='r', linestyle='--', label=f'Mean: {results["mean_length"]:.1f}')
    ax4.set_xlabel('Length (Steps)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Length Distribution')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save validation plots
    validation_graph_file = agent.graph_file.replace('.png', '_validation.png')
    plt.savefig(validation_graph_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Validation plots saved to: {validation_graph_file}")


# Additional utility functions for specific validation scenarios

def validate_agent_continuous(agent, duration_minutes=5, render=False):
    """
    Run continuous validation for a specified duration.
    
    Args:
        agent (Agent): The agent to validate
        duration_minutes (int): Duration to run validation in minutes
        render (bool): Whether to render the environment
    
    Returns:
        dict: Validation results
    """
    import time
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    episode_rewards = []
    episode_lengths = []
    successful_episodes = 0
    episode = 0
    
    agent.setup_environment(render=render)
    agent.setup_networks(for_training=False)
    
    print(f"Running continuous validation for {duration_minutes} minutes...")
    
    try:
        while time.time() < end_time:
            episode_reward, episode_length, success = _run_validation_episode(
                agent, episode, None
            )
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if success:
                successful_episodes += 1
            
            episode += 1
            
            if episode % 10 == 0:
                elapsed = (time.time() - start_time) / 60
                print(f"Episode {episode}, Elapsed: {elapsed:.1f}min, "
                      f"Avg Reward: {np.mean(episode_rewards[-10:]):.2f}")
    
    except KeyboardInterrupt:
        print("Continuous validation interrupted")
    
    finally:
        agent.cleanup()
    
    # Calculate and return results
    results = _calculate_validation_stats(
        episode_rewards, episode_lengths, successful_episodes, episode
    )
    
    _print_validation_summary(results)
    _save_validation_results(agent, results)
    
    return results


# If run directly, provide a simple test
if __name__ == '__main__':
    import yaml
    import os
    from agent import Agent
    
    # Create a simple test configuration
    test_config = {
        'test_val': {
            'env_id': 'CartPole-v1',
            'learning_rate_a': 0.001,
            'discount_factor_g': 0.99,
            'network_sync_rate': 100,
            'replay_memory_size': 1000,
            'mini_batch_size': 32,
            'epsilon_init': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'stop_on_reward': 200,
            'fc1_nodes': 128,
            'enable_double_dqn': True,
            'enable_dueling_dqn': True
        }
    }
    
    # Save test config
    with open('hyperparameters.yml', 'w') as f:
        yaml.dump(test_config, f)
    
    try:
        print("Starting validation test (note: this requires a trained model)...")
        agent = Agent('test_val')
        
        # This will fail without a trained model, but shows the structure
        try:
            results = validate_agent(agent, render=False, num_episodes=3)
            print("Validation test completed successfully!")
        except FileNotFoundError as e:
            print(f"Expected error (no trained model): {e}")
            print("Validation structure test completed - would work with trained model")
    
    finally:
        # Cleanup
        if os.path.exists('hyperparameters.yml'):
            os.remove('hyperparameters.yml')